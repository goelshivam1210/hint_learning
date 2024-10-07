import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from ppo import PPO


# Import environment and wrapper
from env import SimpleEnv
from env_wrapper import EnvWrapper

# import network for attention
from attention_net_own import AttentionNet

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

# Flag for using the wrapper or not
USE_WRAPPER = True  # Change this flag to toggle between wrapped (constraints encoded +concatenated)
USE_ATTENTION = True # Change this flag to toggle between using attention or not

# Utility function to flatten dict observation space for the unwrapped environment
def flatten_obs_space(obs_dict):
    if isinstance(obs_dict, dict) and 'lidar' in obs_dict and 'inventory' in obs_dict:
        lidar_obs = obs_dict['lidar'].flatten()
        inventory_obs = obs_dict['inventory']
        return np.concatenate([lidar_obs, inventory_obs], axis=-1)
    else:
        # If it's already flattened, return it as is
        return obs_dict

# Create a unique identifier for this training instance
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
instance_id = f"ppo_instance_{timestamp}hints_{USE_WRAPPER}_attention_{USE_ATTENTION}"

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
    env = SimpleEnv(render_mode=None)  # Create the base environment
    if USE_WRAPPER:
        wrapped_env = EnvWrapper(env, "constraints.yaml")  # Wrap with constraint handler
        return wrapped_env
    else:
        return env  # Return the base environment without wrapping

env = make_env()
test_env = make_env()
# Observation and action space
dummy_env = make_env()
action_space = dummy_env.action_space

# Handling the observation space shape for wrapped and unwrapped cases
if USE_WRAPPER:
    obs_space = dummy_env.observation_space
    combined_shape = obs_space.shape
else:
    # Flatten observation space from dict to a vector for the unwrapped case
    obs_sample = dummy_env.reset()[0]  # Sample an observation to infer the shape
    flattened_obs = flatten_obs_space(obs_sample)
    combined_shape = flattened_obs.shape  # Get the flattened observation shape

print(f"Observation space shape: {combined_shape}")
print(f"Action space shape: {action_space}")

if USE_ATTENTION:
    if isinstance(dummy_env, EnvWrapper):
        constraint_dim = len(dummy_env.constraints)
        # print (f"Constraint dimension: {constraint_dim}")
    else:
        constraint_dim = 0  # If running without wrapper, constraints don't exist


# PPO setup
ppo_agent = PPO(
    state_dim=combined_shape[0], 
    action_dim=action_space.n, 
    lr_actor=lr_actor, 
    lr_critic=lr_critic,
    gamma=gamma, 
    K_epochs=K_epochs, 
    eps_clip=eps_clip, 
    use_attention=USE_ATTENTION,  # Toggle attention
    attention_net=AttentionNet(state_dim=combined_shape[0], constraint_dim=constraint_dim) if USE_ATTENTION else None  # Pass attention network if needed
)
# Training Loop
def train():
    timestep = 0
    success_train = 0
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()

        cumulative_reward = 0

        # Encode constraints (if using attention)
        constraints = dummy_env.encode_constraints() if USE_ATTENTION else None
        if USE_ATTENTION:
            constraints = torch.tensor(constraints, dtype=torch.float32).to(device)  # Convert to tensor

        for t in range(max_timesteps):
            timestep += 1

            # Select action (pass constraints if using attention)
            action = ppo_agent.select_action(state, constraints=constraints)

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
                success_train += 1

            if terminated or truncated:
                break

        # TensorBoard logging for training
        writer.add_scalar("Train/Reward", cumulative_reward, episode)
        writer.add_scalar("Train/Success_Rate", success_train, episode)

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
# Testing function
def test_agent(env, agent):
    # Clone the policy to ensure no interference between testing and training
    test_agent_policy = PPO(
        state_dim=combined_shape[0], 
        action_dim=action_space.n, 
        lr_actor=lr_actor, 
        lr_critic=lr_critic,
        gamma=gamma, 
        K_epochs=K_epochs, 
        eps_clip=eps_clip, 
        use_attention=USE_ATTENTION,  # Ensure use_attention is passed
        attention_net=AttentionNet(state_dim=combined_shape[0], constraint_dim=constraint_dim) if USE_ATTENTION else None  # Pass attention network if needed
    )
    
    # Use the current policy for testing
    test_agent_policy.policy.load_state_dict(agent.policy.state_dict())

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