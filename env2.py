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


class RewardType(Enum):
    DENSE = "dense"
    SPARSE = "sparse"

class EnvType(Enum):
    PLAIN = "plain"
    CLUTTER = "clutter"

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
        craft_sword = 4

    def __init__(
            self,
            size=12,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            reward_type: RewardType = RewardType.SPARSE,
            env_type: EnvType = EnvType.PLAIN,
            max_steps: int | None = None,
            max_reward_episodes: int = 50,  # Number of episodes with sword reward
            **kwargs,
        ):
            self.step_count = 0
            # self.cumulative_reward = 0
            self.agent_start_pos = agent_start_pos
            self.agent_start_dir = agent_start_dir
            self.reward_type = reward_type
            self.env_type = env_type

            # Ores from which other types of sword can be crafted
            self.clutter_ores = ["gold", "silver", "titanium", "platinum"]
    
            # If env_type is CLUTTER, extend the Actions enum with additional crafting actions
            if self.env_type == EnvType.CLUTTER:
                next_value = len(self.Actions)
                for ore in self.clutter_ores:
                    if f'craft_{ore}_sword' not in self.Actions._member_names_:
                        extend_enum(self.Actions, f'craft_{ore}_sword', next_value)
                        next_value += 1

            self.max_reward_episodes = max_reward_episodes  # Threshold for giving sword reward
            self.current_episode = 0  # Track the current episode

            self.sword_crafted = False

            # Track which resources have been collected during the entire training
            self.collected_resources_global = set()

            # Updated the resource_names to reflect only non-collected world items
            self.resource_names = ["iron_ore", "tree", "crafting_table", "wall"]  # For lidar and inventory
            if self.env_type == EnvType.CLUTTER:
                for ore in self.clutter_ores:
                    if f"{ore}_ore" not in self.resource_names:
                        self.resource_names.append(f"{ore}_ore")
            self.facing_objects = self.resource_names + ["nothing"]  # Include "nothing" for facing logic

            # Inventory for collected items
            self.inventory_items = ["iron", "wood", "iron_sword"]
            if self.env_type == EnvType.CLUTTER:
                for ore in self.clutter_ores:
                    if ore not in self.resource_names:
                        self.inventory_items.append(ore)
                        self.inventory_items.append(f"{ore}_sword")

            self.inventory = []
            mission_space = MissionSpace(mission_func=self._gen_mission)

            # I presume we do not need to track clutter ores
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
            facing_object_shape = len(self.facing_objects)  # One-hot vector for facing object
            total_obs_dim = lidar_shape + inventory_shape + facing_object_shape

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
        if self.env_type == EnvType.CLUTTER:
            for ore in self.clutter_ores:
                self.place_obj(Resource("yellow", f"{ore}_ore"), top=(0, 0))
        self.place_obj(Resource("green", "tree"), top=(0, 0))
        self.place_obj(Box("blue"), top=(0, 0))    # Crafting table

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
        
        # Define angles for each beam relative to the agent's current direction
        base_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)  # Global angles for 8 beams
        agent_angle = self.agent_dir * (np.pi / 2)  # Convert agent_dir to radians (90-degree steps)
        angles = (base_angles + agent_angle) % (2 * np.pi)  # Rotate to align with agent's facing direction

        # Loop through each beam
        for i, angle in enumerate(angles):
            x, y = self.agent_pos  # Start position of the agent
            x_ratio, y_ratio = np.cos(angle), np.sin(angle)

            # Continue shooting the beam until it hits a wall or exits the grid
            for beam_range in range(1, self.grid.width * 2):  # Maximum range based on the grid size
                x_obj = int(x + beam_range * x_ratio)
                y_obj = int(y + beam_range * y_ratio)

                # Check if the beam is out of bounds
                if not (0 <= x_obj < self.grid.width and 0 <= y_obj < self.grid.height):
                    break  # Beam has exited the grid

                # Get the object at the current beam position
                obj = self.grid.get(x_obj, y_obj)
                if isinstance(obj, Floor):
                    self.grid.set(x_obj, y_obj, Floor("yellow"))  # For visualization purposes

                if obj is not None:
                    # Get the index of the detected object
                    if not isinstance(obj, Floor):
                        closest_entity_idx = self.get_entity_index(obj)
                        
                        # Record the distance in the corresponding beam and object type channel
                        if lidar_obs[i, closest_entity_idx] == 0:  # Only update if no previous object was detected on this beam
                            lidar_obs[i, closest_entity_idx] = beam_range / self.grid.width  # Normalize the distance

                    # Stop the beam if the object is a wall (can_occlude)
                    if obj.type == "wall":  # Checking if the object is a wall by its type
                        break  # Stop the beam as it hit a wall

        # **Validate consistency of lidar observations**
        self.validate_lidar_consistency(lidar_obs)

        return lidar_obs
    

    def get_entity_index(self, obj):
        # Map object to the corresponding index in self.resource_names
        if isinstance(obj, Resource):
            return self.resource_names.index(obj.resource_name)
        elif isinstance(obj, Box):
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
        facing_object_one_hot = self.get_facing_object_one_hot()  # Get one-hot encoding for facing object

        # Debugging prints for lidar and inventory shapes
        # print(f"Debug: Lidar Observation Shape: {lidar_obs.shape}")
        # print(f"Debug: Inventory Observation Shape: {inventory_obs.shape}")

        # Concatenate lidar and inventory observations
        combined_obs = np.concatenate([lidar_obs, inventory_obs, facing_object_one_hot], axis=0).astype(np.float32)
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
        reward = -0.1  # Default time step penalty
        # self.cumulative_reward += reward  # Track the cumulative reward
        terminated = False
        truncated = False
        
        self.step_count += 1
        if self.step_count >= self.max_steps:
            terminated = True

        # Action for crafting the sword
        """
        if action == self.Actions.craft_sword.value:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            if isinstance(fwd_cell, Box) and fwd_cell.color == 'blue':  # Crafting table
                if "wood" in self.inventory and "iron" in self.inventory and not self.sword_crafted:
                    self.inventory.remove("wood")
                    self.inventory.remove("iron")
                    self.inventory.append("iron_sword")
                    # print("Crafted an Iron Sword!")
                    self.sword_crafted = True
                    self.crafted_sword_episodes += 1
                    reward = 1000  # Large reward for finding the treasure (sparse reward)
                    terminated = True
        """
        if action == self.Actions.craft_sword.value or (
            self.env_type == EnvType.CLUTTER and 
            any(action == getattr(self.Actions, f'craft_{ore}_sword').value 
                for ore in self.clutter_ores)
        ):
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            if isinstance(fwd_cell, Box) and fwd_cell.color == 'blue':  # Crafting table
                if action == self.Actions.craft_sword.value:
                    if "wood" in self.inventory and "iron" in self.inventory and not self.sword_crafted:
                        self.inventory.remove("wood")
                        self.inventory.remove("iron")
                        self.inventory.append("iron_sword")
                        self.sword_crafted = True
                        self.crafted_sword_episodes += 1
                        reward = 1000
                        terminated = True
                
                elif self.env_type == EnvType.CLUTTER:
                    for ore in self.clutter_ores:
                        if action == getattr(self.Actions, f'craft_{ore}_sword').value:
                            if "wood" in self.inventory and ore in self.inventory:
                                self.inventory.remove("wood")
                                self.inventory.remove(ore)
                                self.inventory.append(f"{ore}_sword")
                            break

            return self.get_obs(), reward, terminated, truncated, {}

        elif action == self.Actions.toggle.value:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)

            if fwd_cell is None:
                return self.get_obs(), reward, terminated, truncated, {}

            # Only allow Resource objects to be collected
            if isinstance(fwd_cell, Resource):
                resource_name = fwd_cell.resource_name
                if resource_name not in self.collected_resources_global:
                    self.collected_resources_global.add(resource_name)
                    # Map the resource to its inventory name
                    if resource_name == "iron_ore":
                        self.inventory.append("iron")
                    elif resource_name == "silver_ore":
                        self.inventory.append("silver")
                    elif resource_name == "gold_ore":
                        self.inventory.append("gold")
                    elif resource_name == "platinum_ore":
                        self.inventory.append("gold")
                    elif resource_name == "titanium_ore": # No harm in having these but could be more modular
                        self.inventory.append("gold")
                    elif resource_name == "tree":
                        self.inventory.append("wood")

                    # Update the lidar observation and inventory
                    self.grid.set(*fwd_pos, None)  # Remove the object from the grid

                    # Assign rewards based on the reward type (dense or sparse)
                    if self.reward_type == RewardType.DENSE:
                        # In DENSE mode, reward immediately upon collecting resources
                        if self.collected_resource_episodes[resource_name] < self.max_reward_episodes:
                            reward = 10  # Higher reward for first few collections
                        else:
                            reward = 1  # Reduced reward after repeated collections
                        # self.cumulative_reward += reward
                    else:
                        # In SPARSE mode, no reward for collecting resources
                        reward = -0.1  # Collecting resources doesn't give immediate rewards in sparse mode
                else:
                    # Penalize redundant collection within the same episode
                    reward = -0.1  
            return self.get_obs(), reward, terminated, truncated, {}

        # Handle basic actions (move, turn, etc.) using the parent class
        if action in [self.Actions.move_forward.value, self.Actions.turn_left.value, self.Actions.turn_right.value]:
            obs, reward_super, terminated, truncated, info = super().step(action)
            self.step_count -= 1
            reward += reward_super
            # self.cumulative_reward += reward  # Update cumulative reward
            return self.get_obs(), reward, terminated, truncated, info

        # Handle unknown actions
        else:
            raise ValueError(f"Unknown action: {action}")
   
    def reset(self, seed=None, **kwargs):
        # print (f"Cumulative reward: {self.cumulative_reward}")
        # print (f"episodes for crafting sword = {self.crafted_sword_episodes}")
        self.np_random, seed = seeding.np_random(seed)
        self.inventory = []
        self.sword_crafted = False  # Reset sword crafting per episode

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
        print (f"Action = {action})")
        print (f"obs = {obs}")
        print(f"step={self.env.step_count}, reward={reward:.2f}")
        # print (f"self.cumulated reward = {self.env.cumulative_reward:.2f}")


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
            "s": SimpleEnv2.Actions.craft_sword.value,  # 'c' for craft sword
            }
        
        # Add any actions in case extended env
        for member in self.env.Actions:
            if member.value not in key_to_action.values():
                print(f"mapping action w/ {chr(member.value + 97)}")
                key_to_action[chr(member.value + 97)] = member.value

        if key in key_to_action:
            print(f"env.inventory: {self.env.inventory}")
            action = key_to_action[key]
            self.step(action)
        else:
            print("Unmapped key:", key)


def main():
    env = SimpleEnv2(render_mode="human", env_type=EnvType.CLUTTER)
    manual_control = CustomManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()