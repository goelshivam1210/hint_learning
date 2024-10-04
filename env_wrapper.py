import numpy as np
from gymnasium import Wrapper
import gymnasium as gym

class EnvWrapper(Wrapper):
    def __init__(self, env, constraints):
        super(EnvWrapper, self).__init__(env)
        self.constraints = constraints

        # Adjust the observation space to account for the augmented state
        original_obs_shape = self.env.observation_space["lidar"].shape[0] + self.env.observation_space["inventory"].shape[0]
        constraint_shape = len(self.constraints)
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(original_obs_shape + constraint_shape,), dtype=np.float32
        )

    def encode_constraints(self):
        """Encode hint constraints as a one-hot vector dynamically."""
        one_hot = np.zeros(len(self.constraints), dtype=np.float32)
        for i, constraint in enumerate(self.constraints):
            if self._check_constraint(constraint):
                one_hot[i] = 1.0
        return one_hot

    def _check_constraint(self, constraint):
        """Check if a constraint is satisfied dynamically, with support for negation."""
        # Check if the constraint contains "not", and adjust parsing accordingly
        is_negated = constraint.startswith("not ")

        # Remove "not " from the constraint if it exists
        if is_negated:
            constraint = constraint.replace("not ", "")

        # Split the constraint into parts: "inventory(item) > 0" -> ["inventory", "item", ">", "0"]
        parts = constraint.split()

        # Handle inventory constraints
        if "inventory" in parts[0]:
            item = parts[0].split("(")[1].replace(")", "")
            operator = parts[1]
            value = int(parts[2])
            item_count = self.env.inventory.count(item)

            # Evaluate the condition based on the operator
            if operator == ">":
                result = item_count > value
            elif operator == "=":
                result = item_count == value
            else:
                result = False

            # Return the negated result if the constraint was negated
            return not result if is_negated else result

        # Handle holding constraints
        elif "holding" in parts[0]:
            item = parts[0].split("(")[1].replace(")", "")
            result = item in self.env.inventory and self.env.inventory[-1] == item  # Check if the agent is holding the specific item

            # Return the negated result if the constraint was negated
            return not result if is_negated else result

        # Handle facing constraints
        elif "facing" in parts[0]:
            target = parts[0].split("(")[1].replace(")", "")
            result = self.env.get_entity_index(target) == self.env.facing

            # Return the negated result if the constraint was negated
            return not result if is_negated else result

        # Default to False if the constraint type is not recognized
        return False

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
        """Augment the current state with the hint constraint encoding."""
        lidar_obs = state["lidar"].flatten().astype(np.float32)
        inventory_obs = state["inventory"].astype(np.float32)
        constraint_encoding = self.encode_constraints()

        # Concatenate the original state with the constraint encoding
        augmented_state = np.concatenate((lidar_obs, inventory_obs, constraint_encoding), axis=-1)
        return augmented_state