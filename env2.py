from __future__ import annotations
import gymnasium as gym
from gymnasium.utils import seeding
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Floor
from minigrid.minigrid_env import MiniGridEnv
from aenum import Enum
from aenum import extend_enum
from minigrid.core.constants import COLOR_NAMES
import pygame
import numpy as np

# for visualization
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()


class RewardType(Enum):
    DENSE = "dense"
    SPARSE = "sparse"

class Resource(Ball):
    """Custom Ball object with a resource name"""
    def __init__(self, color, resource_name):
        super().__init__(color)
        self.resource_name = resource_name

class SimpleEnv2(MiniGridEnv):
    class Actions(Enum):
        move_forward = 0
        turn_left = 1
        turn_right = 2
        toggle = 3
        craft_iron_sword = 4
        craft_copper_sword = 5
        craft_bronze_sword = 6
        craft_silver_sword = 7
        craft_gold_sword = 8
        open_treasure = 9

    def __init__(
            self,
            size=12,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            reward_type: RewardType = RewardType.SPARSE,
            max_steps: int | None = None,
            max_reward_episodes: int = 50,  # Number of episodes with sword reward
            **kwargs,
        ):
            self.step_count = 0
            # self.cumulative_reward = 0
            self.agent_start_pos = agent_start_pos
            self.agent_start_dir = agent_start_dir
            self.reward_type = reward_type

            self.max_reward_episodes = max_reward_episodes  # Threshold for giving sword reward
            self.current_episode = 0  # Track the current episode

            # self.sword_crafted = False
            self.treasure_obtained = False

            # Track which resources have been collected during the entire training
            self.collected_resources_global = set()

            self.useless_items = ["feather", "bone"]  # Items that do nothing

            # resource_names to reflect only non-collected world items
            self.resource_names = ["iron_ore", 
                                   "copper_ore", "bronze_ore", "silver_ore", "gold_ore",
                                    # "tree",
                                    "treasure", "crafting_table", 
                                    "wall"] + self.useless_items # for lidar and inventory
            
            self.inventory_items = ["iron",
                                     "copper", "bronze", "silver", "gold",
                                    # "wood",
                                    "iron_sword", 
                                    "titanium_sword",
                                    "copper_sword", "bronze_sword", "silver_sword", "gold_sword",
                                    "treasure"]+ self.useless_items
            
            # self.facing_objects = self.resource_names + ["nothing"]  # Include "nothing" for facing logic

            # Inventory for collected items
            # self.inventory_items = ["iron", "wood", "iron_sword"]

            self.inventory = ["titanium_sword", "wood", "wood", "wood", "wood", "wood"]  # Agent starts with a titanium sword
            mission_space = MissionSpace(mission_func=self._gen_mission)

            self.crafted_sword_episodes = 0
            self.collected_resource_episodes = {
                "tree": 0,
                "iron_ore": 0,
            }
            # if max_steps is None:
            #     max_steps = 4 * size**2
            if max_steps is None:
                max_steps = 500

            super().__init__(
                mission_space=mission_space,
                grid_size=size,
                see_through_walls=True,
                max_steps=max_steps,
                **kwargs,
            )

            self.action_space = gym.spaces.Discrete(len(self.Actions))
            # print (f"Actions space in the constructor = {self.action_space}")

        # Calculate observation space dimensions
            lidar_shape = 8 * len(self.resource_names)  # Flattened lidar
            inventory_shape = len(self.inventory_items)  # Inventory items
            # facing_object_shape = len(self.facing_objects)  # One-hot vector for facing object
            # total_obs_dim = lidar_shape + inventory_shape + facing_object_shape
            total_obs_dim = lidar_shape + inventory_shape # trying this
            # total_obs_dim = lidar_shape # trying this now


            # Set observation space to a flat Box
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(total_obs_dim,),
                dtype=np.float32
            )

    @staticmethod
    def _gen_mission():
        return "Collect resources, craft a sword, and find the treasure."

    def _gen_grid(self, width, height):
        # Create an empty grid and build the walls
        self.grid = Grid(width, height)

        self.grid.wall_rect(0, 0, width, height)

        for i in range(self.width):
            for j in range(self.width):
                obj = self.grid.get(i, j)
                if isinstance(obj, Floor):
                    self.grid.set(i, j, None)

        # Place resources and other objects on the grid
        self.place_obj(Resource("red", "iron_ore"), top=(0, 0))
        self.place_obj(Resource("blue", "copper_ore"), top=(0, 0))
        self.place_obj(Resource("purple", "bronze_ore"), top=(0, 0))
        self.place_obj(Resource("green", "silver_ore"), top=(0, 0))
        self.place_obj(Resource("yellow", "gold_ore"), top=(0, 0))
        self.place_obj(Resource("grey", "bone"), top=(0, 0))
        self.place_obj(Resource("grey", "feather"), top=(0, 0)) 
        # for _ in range (5):
        #     self.place_obj(Resource("green", "tree"), top=(0, 0))


        self.place_obj(Box("blue"), top=(0, 0))    # Crafting table
        self.place_obj(Box("purple"), top=(0, 0))  # Treasure

        # Ensure the agent's starting position is placed in a valid, empty cell
        if self.agent_start_pos is not None:
            start_cell = self.grid.get(*self.agent_start_pos)
            if start_cell is not None and not start_cell.can_overlap():
                # Find a new valid position for the agent if the start position is occupied
                empty_positions = [
                    (x, y) for x in range(1, width - 1)
                    for y in range(1, height - 1)
                    if self.grid.get(x, y) is None
                ]
                if empty_positions:
                    self.agent_start_pos = empty_positions[0]  # Set to the first empty position
                else:
                    raise RuntimeError("No valid starting position available for the agent.")
            
            # Place the agent in the valid position
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()  # Place the agent randomly if no start position is specified

    def validate_lidar_consistency(self, lidar_obs):

        for beam_idx, beam in enumerate(lidar_obs):
            for obj_idx, distance in enumerate(beam):
                if distance > 0:  # Object detected in this beam
                    expected_obj = self.resource_names[obj_idx]
                    if expected_obj not in self.resource_names:
                        raise ValueError(
                            f"Inconsistent object detected! Beam {beam_idx}, Obj Index {obj_idx}, "
                            f"Detected Obj: {expected_obj} not in resource_names: {self.resource_names}"
                        )
        return True
        
    def get_lidar_observation(self):
        """
        Returns lidar observations aligned with the agent's orientation.
        Each slot corresponds to a beam relative to the agent's facing direction.
        """
        # Initialize the lidar observation matrix: 8 beams x number of object types
        lidar_obs = np.zeros((8, len(self.resource_names)), dtype=np.float32)

        # **Step 1: Reset only previously yellow floors to grey**
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                if isinstance(obj, Floor) and obj.color == "yellow":  # Only reset previously highlighted tiles
                    self.grid.set(x, y, Floor("grey"))  

        # Compute Lidar Observations**
        base_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)  # 8 beam directions
        agent_angle = self.agent_dir * (np.pi / 2)  # Convert agent_dir to radians (90-degree steps)
        angles = (base_angles + agent_angle) % (2 * np.pi)  # Rotate to align with agent's facing direction

        for i, angle in enumerate(angles):
            x, y = self.agent_pos  # Agent's position
            x_ratio, y_ratio = np.cos(angle), np.sin(angle)

            min_distance = float("inf")  # Track closest object distance

            for beam_range in range(1, self.grid.width * 2):  # Max range based on grid size
                x_obj = int(x + beam_range * x_ratio)
                y_obj = int(y + beam_range * y_ratio)

                # Check if the beam is out of bounds
                if not (0 <= x_obj < self.grid.width and 0 <= y_obj < self.grid.height):
                    break  # Beam exits the grid

                obj = self.grid.get(x_obj, y_obj)

                # highlight empty floors, not objects
                if obj is None or isinstance(obj, Floor):
                    self.grid.set(x_obj, y_obj, Floor("yellow"))  # Show beam path
                    continue  # Keep going until an object is found

                # If an object is detected, record the distance
                closest_entity_idx = self.get_entity_index(obj)

                # **Update only if this object is the closest in this beam**
                if beam_range < min_distance:
                    lidar_obs[i, closest_entity_idx] = beam_range / self.grid.width  # Normalize distance
                    min_distance = beam_range  # Update closest object distance

                # **Stop the beam immediately when hitting a wall or object**
                if hasattr(obj, "type") and obj.type == "wall":
                    wall_index = self.resource_names.index("wall")
                    lidar_obs[i, wall_index] = beam_range / self.grid.width
                    break  # Stop at the wall

                break  # Stop at the first detected object

        # **Validate consistency of lidar observations**
        self.validate_lidar_consistency(lidar_obs)

        return lidar_obs
    

    def get_entity_index(self, obj):

        # Map the object to its corresponding index in self.resource_names.
        if isinstance(obj, Resource):
            return self.resource_names.index(obj.resource_name)
        elif isinstance(obj, Box):
            if obj.color == "purple":  # Treasure
                return self.resource_names.index("treasure")
            elif obj.color == "blue":  # Crafting table
                return self.resource_names.index("crafting_table")
        else:
            return self.resource_names.index("wall")

    def get_inventory_observation(self):
        # Updated to only track items that can be added to inventory
        inventory_obs = np.zeros(len(self.inventory_items), dtype=np.float32)
        # print (f"Inventory  = {self.inventory}")
        # print (f"inventory_items = {self.inventory_items}")
        for item in self.inventory:
            if item in self.inventory_items:
                index = self.inventory_items.index(item)
                inventory_obs[index] += 1
        return inventory_obs
    
    def get_facing_object_one_hot(self):
        """
        Get a one-hot encoding representing the object the agent is facing.
        Includes "nothing" if the agent is facing an empty cell.
        Returns:
            np.ndarray: One-hot encoding of the object the agent is facing.
        """
        fwd_pos = self.front_pos  # Position directly in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)  # Get the object at that position

        # Initialize a zeroed one-hot vector
        one_hot = np.zeros(len(self.facing_objects), dtype=np.float32)

        # Determine the index of the object if present
        if fwd_cell is not None:
            if isinstance(fwd_cell, Resource):
                obj_index = self.facing_objects.index(fwd_cell.resource_name)
            elif isinstance(fwd_cell, Box):
                obj_index = self.facing_objects.index("crafting_table")
            else:
                obj_index = self.facing_objects.index("wall")
        else:
            # If no object, set index for "nothing"
            obj_index = self.facing_objects.index("nothing")

        one_hot[obj_index] = 1  # Set the corresponding index to 1

        return one_hot

    def get_obs(self):
        lidar_obs = self.get_lidar_observation().flatten().astype(np.float32)   # Flatten lidar
        inventory_obs = self.get_inventory_observation().astype(np.float32)
        # facing_object_one_hot = self.get_facing_object_one_hot()  # Get one-hot encoding for facing object

        # Debugging prints for lidar and inventory shapes
        # print(f"Debug: Lidar Observation Shape: {lidar_obs.shape}")
        # print(f"Debug: Inventory Observation Shape: {inventory_obs.shape}")

        # Concatenate lidar and inventory observations
        # combined_obs = np.concatenate([lidar_obs, inventory_obs, facing_object_one_hot], axis=0).astype(np.float32)
        combined_obs = np.concatenate([lidar_obs, inventory_obs], axis=0).astype(np.float32)
        # combined_obs = np.concatenate([lidar_obs], axis=0).astype(np.float32)

        # print(f"Debug: Combined Observation Shape: {combined_obs.shape}")

        return combined_obs
    
    # utility function to find path to object
    def find_object_position(self, obj_name):
        """Find the position of the object in the grid by its name."""
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                if isinstance(obj, Resource) and obj.resource_name == obj_name:
                    return (x, y)
                elif isinstance(obj, Box) and obj_name == "chest" and obj.color == "purple":
                    return (x, y)
                elif isinstance(obj, Box) and obj_name == "crafting_table" and obj.color == "blue":
                    return (x, y)
        return None
    def get_adjacent_pos_and_dir(self, agent_pos, target_pos):
        """Given the agent's position and the target position, find the adjacent position and direction to face the object."""
        x, y = target_pos

        # List of adjacent positions and the direction the agent should face to look at the target object
        adjacent_positions = [
            ((x - 1, y), 0),  # Left of the object, face right
            ((x + 1, y), 2),  # Right of the object, face left
            ((x, y - 1), 3),  # Below the object, face up
            ((x, y + 1), 1),  # Above the object, face down
        ]

        # Loop through the possible adjacent positions
        for adj_pos, adj_dir in adjacent_positions:
            # Ensure the adjacent cell is valid (empty or movable) and within bounds
            if 0 <= adj_pos[0] < self.grid.width and 0 <= adj_pos[1] < self.grid.height:
                adj_cell = self.grid.get(*adj_pos)
                if adj_cell is None:  # Ensure the adjacent cell is empty
                    return adj_pos, adj_dir

        return None, None  # Return None if no valid adjacent position is found
    
    def step(self, action):
        reward = -1  # Default time step penalty
        terminated = False
        truncated = False
        
        self.step_count += 1
        if self.step_count >= self.max_steps:
            terminated = True

        if action in [self.Actions.craft_iron_sword.value,
                        self.Actions.craft_copper_sword.value,
                        self.Actions.craft_bronze_sword.value,
                        self.Actions.craft_silver_sword.value,
                        self.Actions.craft_gold_sword.value
                        ]:
            
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            if isinstance(fwd_cell, Box) and fwd_cell.color == 'blue':  # Crafting table check
                ore_type = {
                    self.Actions.craft_iron_sword.value: "iron",
                    self.Actions.craft_copper_sword.value: "copper",
                    self.Actions.craft_bronze_sword.value: "bronze",
                    self.Actions.craft_silver_sword.value: "silver",
                    self.Actions.craft_gold_sword.value: "gold"
                }.get(action, None)

                if ore_type and ore_type in self.inventory and "wood" in self.inventory:
                    # print(f"Before crafting: Inventory = {self.inventory}")
                    self.inventory.remove(ore_type)
                    self.inventory.remove("wood")
                    crafted_sword = f"{ore_type}_sword"
                    self.inventory.append(crafted_sword)
                    # print(f"After crafting: Inventory = {self.inventory}")
                    # print(f"Crafted {crafted_sword}!")

            return self.get_obs(), reward, terminated, truncated, {}

        # === Open Treasure (Only Works with Iron Sword) ===
        elif action == self.Actions.open_treasure.value:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            if isinstance(fwd_cell, Box) and fwd_cell.color == "purple":  # Treasure Box
                if "iron_sword" in self.inventory:  # Must be iron_sword
                    self.inventory.append("treasure")
                    self.grid.set(*fwd_pos, None)  # Remove treasure from grid
                    # print("Treasure obtained!")
                    reward = 600  # Large reward for success
                    terminated = True
                    truncated = True  # Episode should stop immediately
                    self.treasure_obtained = True

            return self.get_obs(), reward, terminated, truncated, {}

        # === Collect Resources (Includes Useless Items) ===
        elif action == self.Actions.toggle.value:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            if fwd_cell is None:
                return self.get_obs(), reward, terminated, truncated, {}

            if isinstance(fwd_cell, Resource):
                resource_name = fwd_cell.resource_name

                # Assign resource to inventory (Handles ores, trees, and useless objects)
                if resource_name in ["iron_ore", "copper_ore", "bronze_ore", "silver_ore", "gold_ore"]:
                    self.inventory.append(resource_name.replace("_ore", ""))  # Convert eg: "iron_ore" -> "iron"
                elif resource_name == "tree":
                    self.inventory.append("wood")
                elif resource_name in ["feather", "bone"]:  # New useless objects
                    self.inventory.append(resource_name)

                self.grid.set(*fwd_pos, None)  # Remove object from grid

                # Assign rewards based on reward type
                if self.reward_type == RewardType.DENSE:
                    reward = 10 if self.collected_resource_episodes[resource_name] < self.max_reward_episodes else 1
                else:
                    reward = -0.1  # Small penalty for resource collection in sparse mode

            return self.get_obs(), reward, terminated, truncated, {}

        # === Movement and Rotation ===
        if action in [self.Actions.move_forward.value, self.Actions.turn_left.value, self.Actions.turn_right.value]:
            obs, reward_super, terminated, truncated, info = super().step(action)
            self.step_count -= 1  # Adjust for step increase earlier
            reward += reward_super
            return self.get_obs(), reward, terminated, truncated, info

        # Handle unknown actions
        else:
            raise ValueError(f"Unknown action: {action}")
   
    def reset(self, seed=None, **kwargs):
        # print (f"Cumulative reward: {self.cumulative_reward}")
        # print (f"episodes for crafting sword = {self.crafted_sword_episodes}")
        self.np_random, seed = seeding.np_random(seed)
        self.inventory = ["titanium_sword", "wood", "wood", "wood", "wood", "wood" ]  # Agent starts with a titanium sword
        self.sword_crafted = False  # Reset sword crafting per episode
        self.treasure_obtained = False
        self.collected_resources_global = set()

        # Reset per-episode resource collection
        self.collected_resources_global = set()

        # Increase the episode count
        self.current_episode += 1
        # self.cumulative_reward = 0

        self._gen_grid(self.width, self.height)
        self.place_agent()
        for i in range(self.width):
            for j in range(self.width):
                obj = self.grid.get(i, j)
                if obj is None:
                    self.grid.set(i, j, Floor("grey"))

        if self.agent_dir is None:
            self.agent_dir = 0  # Default to facing north if not specified

        self.step_count = 0
        return self.get_obs(), {}
    
    def render(self):
        # Call the parent class's render method
        result = super().render()

        # Display the agent's inventory on the screen (in the terminal or add GUI)
        # print(f"Inventory: {', '.join(self.inventory)}")

        return result

    def debug_print_observation(self, obs):
        """Nicely formatted observation print with colors and table layout."""
        lidar_len = 8 * len(self.resource_names)  # Lidar section length
        inventory_len = len(self.inventory_items)  # Inventory section length

        # Split observation into sections
        lidar_obs = obs[:lidar_len].reshape(8, -1)  # 8 lidar beams
        inventory_obs = obs[lidar_len:lidar_len + inventory_len]

        # 🟢 Create Lidar Table
        lidar_table = Table(title="🔍 Lidar Observation (8 Beams x Object Types)", show_lines=True)
        lidar_table.add_column("Beam", style="cyan", justify="center")
        lidar_table.add_column("Detected Objects", style="magenta", justify="left")
        lidar_table.add_column("Vector", style="yellow", justify="left")

        for i, beam in enumerate(lidar_obs):
            labeled_beam = [f"[cyan]{self.resource_names[j]}:[yellow]{beam[j]:.2f}[/yellow] (slot {j})" 
                            for j in range(len(self.resource_names)) if beam[j] > 0]
            lidar_table.add_row(f"[bold green]Beam {i+1}[/bold green]",
                                ", ".join(labeled_beam) if labeled_beam else "[dim]No objects detected[/dim]",
                                f"{beam}")

        # 🟡 Create Inventory Table
        inventory_table = Table(title="🎒 Inventory", show_lines=True)
        inventory_table.add_column("Item", style="green", justify="left")
        inventory_table.add_column("Count", style="bold yellow", justify="center")
        inventory_table.add_column("Slot", style="blue", justify="center")

        inventory_vector = np.zeros(len(self.inventory_items), dtype=np.float32)  # Empty vector
        for i, item in enumerate(self.inventory_items):
            if inventory_obs[i] > 0:
                inventory_vector[i] = inventory_obs[i]
                inventory_table.add_row(f"[bold cyan]{item}[/bold cyan]", f"{int(inventory_obs[i])}", f"[blue]{i}[/blue]")

        # Print everything beautifully!
        console.print(Panel(lidar_table, title="[bold red]LIDAR OBSERVATION[/bold red]", expand=False))
        console.print(Panel(inventory_table, title="[bold blue]INVENTORY[/bold blue]", expand=False))
        console.print(Panel(f"Inventory Vector: {inventory_vector}", title="📊 Inventory Vector", expand=False))


# Custom manual control class for handling custom actions
class CustomManualControl:
    def __init__(self, env, seed=None):
        self.env = env
        self.seed = seed
        self.closed = False

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        print(f"\nAction = {action} ({self.env.Actions(action).name})")
        self.env.debug_print_observation(obs)  # Call debug print
        print(f"Step={self.env.step_count}, Reward={reward:.2f}")


        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()


    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": SimpleEnv2.Actions.turn_left.value,
            "up": SimpleEnv2.Actions.turn_right.value,
            "right": SimpleEnv2.Actions.move_forward.value,
            "space": SimpleEnv2.Actions.toggle.value,
            "o": SimpleEnv2.Actions.open_treasure.value,
            "1": SimpleEnv2.Actions.craft_iron_sword.value,
            "2": SimpleEnv2.Actions.craft_copper_sword.value,
            "3": SimpleEnv2.Actions.craft_bronze_sword.value,
            "4": SimpleEnv2.Actions.craft_silver_sword.value,
            "5": SimpleEnv2.Actions.craft_gold_sword.value
        }
        
        for member in self.env.Actions:
            if member.value not in key_to_action.values():
                print(f"printing {chr(member.value + 97)}")
                key_to_action[chr(member.value + 97)] = member.value

        if key in key_to_action:
            action = key_to_action[key]
            self.step(action)
        else:
            print("Unmapped key:", key)


def main():
    env = SimpleEnv2(render_mode="human")
    manual_control = CustomManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()