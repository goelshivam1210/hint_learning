import numpy as np
from gymnasium import Wrapper
import gymnasium as gym
import yaml

class EnvWrapper(Wrapper):
    def __init__(self, env):
        super(EnvWrapper, self).__init__(env)

        # Identify the index of "iron_sword" in the inventory vector
        self.iron_sword_index = self.env.inventory_items.index("iron_sword")

        # The hint vector is fixed: one-hot encoding where only "iron_sword" is 1
        self.hint_vector = np.zeros(len(self.env.inventory_items), dtype=np.float32)
        self.hint_vector[self.iron_sword_index] = 1.0  # Set the iron_sword slot to 1

        # Compute the total observation shape
        total_obs_shape = self.env.observation_space.shape[0] + len(self.hint_vector)

        # Update observation space to account for the additional hint vector
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_shape,), dtype=np.float32
        )

    def reset(self, **kwargs):
        """Reset the environment and append the fixed hint vector."""
        state, info = self.env.reset(**kwargs)
        augmented_state = self._augment_state_with_hint(state)
        return augmented_state, info

    def step(self, action):
        """Take a step in the environment and append the fixed hint vector."""
        next_state, reward, done, truncated, info = self.env.step(action)
        augmented_state = self._augment_state_with_hint(next_state)
        return augmented_state, reward, done, truncated, info

    def _augment_state_with_hint(self, state):
        """Append the fixed iron_sword hint vector to the state."""
        return np.concatenate((state, self.hint_vector), axis=-1)