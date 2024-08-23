import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from env import SimpleEnv
from ppo import PPO

# Hyperparameters
lr_actor = 0.0003
lr_critic = 0.001
gamma = 0.99
K_epochs = 80
eps_clip = 0.2
max_episodes = 100000
max_timesteps = 300
update_timestep = 2000
save_interval = 1000
log_interval = 10
test_interval = 20  # How often (in episodes) to run tests
n_test_episodes = 20  # Number of test episodes
batch_size = 2000

# Convergence parameters
convergence_threshold = 0.9  # 90% success rate
convergence_window = 10  # Consider last 10 evaluations for success tracking

# Device setup
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a unique identifier for this training instance
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
instance_id = f"ppo_instance_{timestamp}"

# Create a directory for saving models and logs for this training instance
base_dir = os.path.join("log", instance_id)
os.makedirs(base_dir, exist_ok=True)

# Model save directory
model_save_dir = os.path.join(base_dir, "models")
os.makedirs(model_save_dir, exist_ok=True)

# TensorBoard setup
log_dir = os.path.join(base_dir, "logs")
writer = SummaryWriter(log_dir)

# Environment setup
def make_env():
    return SimpleEnv(render_mode=None)

env = make_env()
test_env = make_env()

# Observation and action space
obs_space = env.observation_space
action_space = env.action_space

# Extract observation space for lidar and inventory
lidar_shape = obs_space['lidar'].shape
inventory_shape = obs_space['inventory'].shape
combined_shape = lidar_shape[0] * lidar_shape[1] + inventory_shape[0]

# PPO setup
ppo_agent = PPO(state_dim=combined_shape, action_dim=action_space.n, lr_actor=lr_actor, lr_critic=lr_critic,
                gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip)

# Training Loop
def train():
    timestep = 0
    success_train = 0
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()

        cumulative_reward = 0


        for t in range(max_timesteps):
            timestep += 1

            # Select action
            action = ppo_agent.select_action(state)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Store transition
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(terminated or truncated)

            # Move to the next state
            state = next_state
            cumulative_reward += reward

            # PPO update
            if timestep % update_timestep == 0:
                ppo_agent.update()
                timestep = 0
            
            if terminated:
                success_train +=1

            if terminated or truncated:

                break

        # TensorBoard logging for training
        writer.add_scalar("Train/Reward", cumulative_reward, episode)
        writer.add_scalar("Train/Sucess_Rate", success_train, episode)

        # Print and log
        if episode % log_interval == 0:
            print(f"Episode {episode}/{max_episodes}, Reward: {cumulative_reward:.2f}")

        # Save model periodically
        if episode % save_interval == 0:
            model_path = os.path.join(model_save_dir, f"ppo_model_{episode}.pth")
            ppo_agent.save(model_path)
            print(f"Model saved at episode {episode} to {model_path}")

        # Test the agent periodically
        if episode % test_interval == 0:
            test_rewards, success_rate = test_agent(test_env, ppo_agent)
            writer.add_scalar("Test/Reward", np.mean(test_rewards), episode)
            writer.add_scalar("Test/Success_Rate", success_rate, episode)
            print(f"Test Results - Episode {episode}: Avg Reward: {np.mean(test_rewards):.2f}, Success Rate: {success_rate * 100:.2f}%")

    writer.close()


# Testing function
def test_agent(env, agent):
    # Clone the policy to ensure no interference between testing and training
    test_agent_policy = PPO(state_dim=combined_shape, action_dim=action_space.n, lr_actor=lr_actor, lr_critic=lr_critic,
                            gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip)
    test_agent_policy.policy.load_state_dict(agent.policy.state_dict())  # Use the current policy for testing

    rewards = []
    successes = 0

    for _ in range(n_test_episodes):
        state, _ = env.reset()

        cumulative_reward = 0
        terminated = False

        while not terminated:
            action = test_agent_policy.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            cumulative_reward += reward

            if terminated or truncated:
                break

        rewards.append(cumulative_reward)

        # Assume success is determined by finding the treasure
        if "treasure" in env.inventory:
            successes += 1

    success_rate = successes / n_test_episodes
    return rewards, success_rate


if __name__ == "__main__":
    train()