from __future__ import annotations
import gym
import numpy as np
from env import SimpleEnv  # Import your environment

def test_observation_space(env):
    # Reset the environment to generate initial observation
    obs, _ = env.reset()  # Extract the observation from the reset tuple

    # Debug: Print the structure of the observation
    print("Observation structure:", obs)

    # Check if 'inventory' and 'lidar' keys exist in the observation
    if "inventory" not in obs or "lidar" not in obs:
        raise KeyError("Observation does not contain expected 'inventory' and 'lidar' keys.")

    # Check Inventory Observation
    inventory_obs = obs["inventory"]
    lidar_obs = obs["lidar"]

    # Print out the sizes and dimensions
    print(f"Lidar Observation Shape: {lidar_obs.shape}")
    print(f"Inventory Observation Shape: {inventory_obs.shape}")
    
    # Print the observation data for inspection
    print(f"Lidar Observation:\n{lidar_obs}")
    print(f"Inventory Observation:\n{inventory_obs}")

    # Check that the shapes are correct
    assert isinstance(inventory_obs, np.ndarray), "Inventory observation must be a numpy array."
    assert inventory_obs.shape == (len(env.resource_names),), f"Inventory observation shape is incorrect. Expected: {len(env.resource_names)}, Got: {inventory_obs.shape}"

    # Ensure inventory values are within the expected range (0-10 items)
    assert np.all((inventory_obs >= 0) & (inventory_obs <= 10)), "Inventory observation contains invalid counts."

    print("Inventory observation is valid.")

    # Check LiDAR Observation
    num_beams = 8
    num_entities = len(env.resource_names)

    assert isinstance(lidar_obs, np.ndarray), "LiDAR observation must be a numpy array."
    assert lidar_obs.shape == (num_beams, num_entities), f"LiDAR observation shape is incorrect. Expected: {(num_beams, num_entities)}, Got: {lidar_obs.shape}"

    # Ensure LiDAR distances are valid (0 <= distance <= 1 after normalization by grid width)
    assert np.all((lidar_obs[:, :] >= 0) & (lidar_obs[:, :] <= 1)), "LiDAR observation contains invalid distances."

    print("LiDAR observation is valid.")

    # Ensure each beam has only one detected object type with a non-zero distance
    for i in range(num_beams):
        detected_entities = np.where(lidar_obs[i] > 0)[0]
        assert len(detected_entities) <= 1, f"Multiple objects detected by the same beam: {detected_entities}"

    print("LiDAR beam observation is valid.")

    return True


if __name__ == "__main__":
    env = SimpleEnv(render_mode="rgb_array")
    
    # Test the observation space and print sizes and data
    if test_observation_space(env):
        print("Observation space test passed!")