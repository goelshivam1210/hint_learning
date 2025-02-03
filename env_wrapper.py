import numpy as np
from gymnasium import Wrapper
import gymnasium as gym
import yaml
from env import Resource

class EnvWrapper(Wrapper):
    def __init__(self, env, constraint_file):
        super(EnvWrapper, self).__init__(env)
        self.constraints = self.load_constraints(constraint_file)

        # print ("self.env.observation_space = {}".format(self.env.observation_space.shape[0]))
        constraint_shape = len(self.constraints)

        # Correct the total observation shape: lidar (flattened) + inventory + constraints
        total_obs_shape = self.env.observation_space.shape[0] + constraint_shape
        
        # Update observation space to account for the flattened shape and constraints
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_shape,), dtype=np.float32
        )

    def load_constraints(self, filepath):
        """Load the constraints from a YAML file."""
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
            return data['hint_constraints']

    def encode_constraints(self):
        """Encode hint constraints as a one-hot vector dynamically."""
        one_hot = np.zeros(len(self.constraints), dtype=np.float32)
        for i, constraint in enumerate(self.constraints):
            if self._check_constraint(constraint):
                one_hot[i] = 1.0
        return one_hot

    def _check_constraint(self, constraint):
        """Check if a constraint is satisfied dynamically, with support for negation."""
        is_negated = constraint.startswith("not ")

        # Remove "not " if it exists
        if is_negated:
            constraint = constraint.replace("not ", "")

        parts = constraint.split()

        # Handle inventory constraints
        if "inventory" in parts[0]:
            item = parts[0].split("(")[1].replace(")", "")
            operator = parts[1]
            value = int(parts[2])
            item_count = self.env.inventory.count(item)

            if operator == ">":
                result = item_count > value
            elif operator == "=":
                result = item_count == value
            else:
                result = False

            return not result if is_negated else result

        # Handle holding constraints
        elif "holding" in parts[0]:
            item = parts[0].split("(")[1].replace(")", "")
            result = item in self.env.inventory and self.env.inventory[-1] == item
            return not result if is_negated else result

        # Handle facing constraints
        elif "facing" in parts[0]:
            target = parts[0].split("(")[1].replace(")", "")
            
            # Check for agent_dir initialization
            if self.env.agent_dir is None:
                return False  # Avoid calculating direction if it's not initialized

            # Proceed if direction is valid
            fwd_pos = self.env.front_pos
            obj_in_front = self.env.grid.get(*fwd_pos)
            if obj_in_front is not None and isinstance(obj_in_front, Resource) and obj_in_front.resource_name == target:
                result = True
            else:
                result = False

            return not result if is_negated else result

        return False  # Default case

    def reset(self, **kwargs):
        """Reset the environment and augment the initial state with constraints."""
        state, info = self.env.reset(**kwargs)
        augmented_state = self._augment_state_with_constraints(state)
        return augmented_state, info

    def step(self, action):
        """Take a step in the environment and augment the state with constraints."""
        next_state, reward, done, truncated, info = self.env.step(action)
        augmented_state = self._augment_state_with_constraints(next_state)
        return augmented_state, reward, done, truncated, info

    def _augment_state_with_constraints(self, state):
        # Total lidar, inventory, and facing dimensions from environment
        lidar_len = 8 * len(self.env.resource_names)       # 32
        inventory_len = len(self.env.inventory_items)      # 3
        # facing_len = len(self.env.facing_objects)          # 5

        # Slicing the flat state array
        lidar_obs = state[:lidar_len].flatten().astype(np.float32)  # First 32
        inventory_obs = state[lidar_len:lidar_len + inventory_len].astype(np.float32)  # Next 3
        # facing_obs = state[lidar_len + inventory_len:lidar_len + inventory_len + facing_len].astype(np.float32)  # Next 5

        # Encode constraints as a one-hot vector
        constraint_encoding = self.encode_constraints()
        # print(f"Debug: Constraints Encoding Shape: {constraint_encoding.shape}, Values: {constraint_encoding}")
        # print(f"Debug: Lidar Obs Shape: {lidar_obs.shape}, Inventory Obs Shape: {inventory_obs.shape}, Facing Obs Shape: {facing_obs.shape}")

        # Concatenate the original state (lidar, inventory, facing) with the constraint encoding
        # augmented_state = np.concatenate((lidar_obs, inventory_obs, facing_obs, constraint_encoding), axis=-1)
        augmented_state = np.concatenate((lidar_obs, inventory_obs, constraint_encoding), axis=-1)
        # augmented_state = np.concatenate((lidar_obs, constraint_encoding), axis=-1)

        # print(f"Debug: Augmented State Shape: {augmented_state.shape}")

        return augmented_state