import os
import random
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import argparse
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter
from ppo import PPO
import yaml

# Import environment and wrapper
from editedenv import SimpleEnv
from env2 import SimpleEnv2, RewardType

from env_wrapper import EnvWrapper

# Import attention network
from attention_net_own import AttentionNet

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO with optional attention and environment wrapping.')
    # Environment and attention flags
    parser.add_argument('--use-wrapper', action='store_true', help='Use environment wrapper with constraint encoding.')
    parser.add_argument('--use-attention', action='store_true', help='Use attention mechanism in PPO.')
    parser.add_argument('--use-smallenv', action='store_true', help='Use environment with smaller action space')
    parser.add_argument('--use-dense', action='store_true', help='Use dense reward function')

    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'], help='Device to run the model on.')

    # Hyperparameters
    parser.add_argument('--lr-actor', type=float, default=3e-4, help='Learning rate for the actor.')
    parser.add_argument('--lr-critic', type=float, default=1e-3, help='Learning rate for the critic.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for PPO.')
    parser.add_argument('--K-epochs', type=int, default=5, help='Number of PPO epochs per update.')
    parser.add_argument('--eps-clip', type=float, default=0.2, help='Clip range for PPO updates.')
    parser.add_argument('--grid_size', type=int, default=12, help='Size of the gridworld')
    parser.add_argument('--max_episodes', type=int, default=30000, help='Maximum number of training episodes.')
    parser.add_argument('--max_timesteps', type=int, default=100, help='Maximum number of timesteps per episode.')
    parser.add_argument('--update_timestep', type=int, default=2000, help='Timesteps after which PPO update is triggered.')
    parser.add_argument('--batch_size', type=int, default=128, help="how many collected timesteps (from the environment rollouts) are used in one gradient update.")
    parser.add_argument('--save_interval', type=int, default=10000, help='Interval to save the model.')
    parser.add_argument('--log_interval', type=int, default=1000, help='Interval to log training progress.')
    parser.add_argument('--test_interval', type=int, default=500, help='Interval to test the agent.')
    parser.add_argument('--n_test_episodes', type=int, default=25, help='Number of test episodes.')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 9), help='Random seed for reproducibility.')
    parser.add_argument('--convergence', type=int, default=15, help='Random seed for reproducibility.')

    args = parser.parse_args()
    return args

def set_seed(seed):
    """
    Set the random seed for Python, NumPy, PyTorch, and CUDA.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Global seed set to: {seed}")


# Utility function to flatten dict observation space for the unwrapped environment
def flatten_obs_space(obs_dict):
    if isinstance(obs_dict, dict) and 'lidar' in obs_dict and 'inventory' in obs_dict:
        lidar_obs = obs_dict['lidar'].flatten()
        inventory_obs = obs_dict['inventory']
        return np.concatenate([lidar_obs, inventory_obs], axis=-1)
    else:
        # If it's already flattened, return it as is
        return obs_dict

def get_network_architecture(model):
    """Extract network architecture from a PyTorch model."""
    architecture = []
    for layer in model:
        layer_info = {"layer_type": type(layer).__name__}
        if isinstance(layer, nn.Linear):
            layer_info["in_features"] = int(layer.in_features)  # Python int
            layer_info["out_features"] = int(layer.out_features)  # Python int
        elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.Tanh):
            layer_info["activation"] = type(layer).__name__
        elif isinstance(layer, nn.Softmax):
            layer_info["dim"] = int(layer.dim)  # Python int
        # Add other layer types as needed
        architecture.append(layer_info)
    return architecture

# Main function
def main():
    args = parse_args()
    
    # Set global seed
    set_seed(args.seed)

    # Set device
    device = torch.device(args.device)

    # Create a unique identifier for this training instance
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    instance_id = f"ppo_instance_{timestamp}_hints_{args.use_wrapper}_attention_{args.use_attention}"

    # Create a directory for saving models and logs for this training instance
    base_dir = os.path.join("log", instance_id)
    os.makedirs(base_dir, exist_ok=True)

    # Model save directory
    model_save_dir = os.path.join(base_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)

    # TensorBoard setup
    log_dir = os.path.join(base_dir, "logs")
    writer = SummaryWriter(log_dir)

    # # Sample data (Python dictionary)
    # param_dict = vars(args)

    # with open(os.path.join(base_dir, "params.yaml"), 'w') as file:
    #     yaml.dump(param_dict, file)

    # Environment setup
    def make_env(seed=None):
        if args.use_smallenv:
            env = SimpleEnv2(
                render_mode=None, 
                max_steps=args.max_timesteps, 
                reward_type=RewardType.DENSE if args.use_dense else RewardType.SPARSE, 
                size=args.grid_size
            )
        else:
            env = SimpleEnv(
                render_mode=None, 
                max_steps=args.max_timesteps, 
                reward_type=RewardType.DENSE if args.use_dense else RewardType.SPARSE, 
                size=args.grid_size
            )
        
        if seed is not None:
            env.reset(seed=seed)

        if args.use_wrapper:
            wrapped_env = EnvWrapper(env, "constraints.yaml")
            return wrapped_env
        else:
            return env
        
    # Environment setup for video recording
    def make_env_for_video():
        if args.use_smallenv:
            env = SimpleEnv2(render_mode='rgb_array', max_steps=args.max_timesteps, reward_type=RewardType.DENSE if args.use_dense else RewardType.SPARSE, size=args.grid_size)  # Create the base environment
        else:
            env = SimpleEnv(render_mode='rgb_array', max_steps=args.max_timesteps, reward_type=RewardType.DENSE if args.use_dense else RewardType.SPARSE, size=args.grid_size)  # Enable rgb_array mode for video recording
        if args.use_wrapper:
            wrapped_env = EnvWrapper(env, "constraints.yaml")
            return wrapped_env
        else:
            return env

    # Create environments
    env = make_env(seed=args.seed)
    test_env = make_env(seed=args.seed)
    dummy_env = make_env(seed=args.seed)
    action_space = dummy_env.action_space

    # Handle observation space shape for wrapped and unwrapped cases
    if args.use_wrapper:
        obs_space = dummy_env.observation_space
        combined_shape = obs_space.shape
    else:
        # Flatten observation space from dict to a vector for the unwrapped case
        obs_sample = dummy_env.reset()[0]
        flattened_obs = flatten_obs_space(obs_sample)
        combined_shape = flattened_obs.shape  # Get the flattened observation shape

    print(f"Observation space shape: {combined_shape}")
    print(f"Action space shape: {action_space}")

    if args.use_attention:
        if isinstance(dummy_env, EnvWrapper):
            constraint_dim = len(dummy_env.constraints)
        else:
            constraint_dim = 0  # If running without wrapper, constraints don't exist

    # PPO setup
    ppo_agent = PPO(
        state_dim=combined_shape[0],
        action_dim=action_space.n,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        K_epochs=args.K_epochs,
        eps_clip=args.eps_clip,
        use_attention=args.use_attention,
        attention_net=AttentionNet(state_dim=combined_shape[0], constraint_dim=constraint_dim) if args.use_attention else None
    )

    # Extract actor and critic architectures
    actor_architecture = get_network_architecture(ppo_agent.policy.actor)
    critic_architecture = get_network_architecture(ppo_agent.policy.critic)

    # Add the architectures to the parameter dictionary
    param_dict = vars(args)  # Use the arguments as base parameters
    param_dict["actor_network"] = actor_architecture
    param_dict["critic_network"] = critic_architecture

    # Save the parameters and architectures to the YAML file
    with open(os.path.join(base_dir, "params.yaml"), 'w') as file:
        yaml.dump(param_dict, file)

    # Training Loop
    def train():
        timestep = 0
        success_train = 0
        success_window = []
        converged = False
        for episode in range(1, args.max_episodes + 1):
            state, _ = env.reset()
            state = (state - state.mean()) / state.std()

            cumulative_reward = 0

            # # Encode constraints (if using attention)
            # constraints = dummy_env.encode_constraints() if args.use_attention else None
            # if args.use_attention:
            #     constraints = torch.tensor(constraints, dtype=torch.float32).to(device)  # Convert to tensor

            for t_step in range(args.max_timesteps):
                timestep += 1

                # Select action (pass constraints if using attention)
                if args.use_attention:
                    constraints = dummy_env.encode_constraints()
                    constraints = torch.tensor(constraints, dtype=torch.float32).to(device)
                else:
                    constraints = None

                # Select action (pass constraints if using attention)
                action = ppo_agent.select_action(state, constraints=constraints)

                # Step environment
                next_state, reward, terminated, truncated, _ = env.step(action)

                next_state = (next_state - next_state.mean()) / next_state.std()
                # if reward != -0.1:
                #     print(reward)

                # Store transition
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(terminated or truncated)

                # Move to the next state
                state = next_state
                cumulative_reward += reward

                # PPO update
                if timestep % args.update_timestep == 0:
                    ppo_agent.update()
                    timestep = 0

                if args.use_smallenv:
                    if "iron_sword" in env.inventory:
                        # print ("Goal state reached!")
                        success_train += 1
                else:
                    if "treasure" in env.inventory:
                        success_train += 1

                if terminated:
                    # print("terminated")
                    # print(t)
                    break
                if truncated:
                    # print("truncated")
                    # print(t)
                    break

            # TensorBoard logging for training
            writer.add_scalar("Train/Reward", cumulative_reward, episode)
            writer.add_scalar("Train/Success_Rate", success_train / episode, episode)

            # Print and log
            if episode % args.log_interval == 0:
                print(f"Episode {episode}/{args.max_episodes}, Reward: {cumulative_reward:.2f}")

            # Save model periodically
            if episode % args.save_interval == 0:
                model_path = os.path.join(model_save_dir, f"ppo_model_{episode}.pth")
                ppo_agent.save(model_path)
                print(f"Model saved at episode {episode} to {model_path}")

            # Test the agent periodically
            if episode % args.test_interval == 0:
                test_rewards, success_rate = test_agent(test_env, ppo_agent)
                # Append the success rate to the sliding window
                success_window.append(success_rate)

                # Maintain the window size (last 10 test results)
                if len(success_window) > args.convergence:
                    success_window.pop(0)

                # Check for convergence: if the average success rate over the last 10 tests is > 95%
                if len(success_window) == args.convergence and np.mean(success_window) >= 0.95:
                    print(f"Converged with success rate: {np.mean(success_window):.2f}")
                    converged = True

                writer.add_scalar("Test/Reward", np.mean(test_rewards), episode)
                writer.add_scalar("Test/Success_Rate", success_rate, episode)
                print(f"Test Results - Episode {episode}: Avg Reward: {np.mean(test_rewards):.2f}, Success Rate: {success_rate * 100:.2f}%")
                # Stop training if converged
                if converged:
                    model_path = os.path.join(model_save_dir, f"ppo_model_converged.pth")
                    ppo_agent.save(model_path)
                    print(f"Converged and saved model at {model_path}")
                    break
        # Directory to save the video
        video_dir = os.path.join(base_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

        # Set the episode trigger to only save video for the first 5 episodes
        video_env = RecordVideo(make_env_for_video(), video_dir, episode_trigger=lambda x: x < 5)  # Save videos for episodes 1-5

        # Loop through episodes for video recording (limited to the first 5)
        for episode in range(1, 6):  # Loop only for 5 episodes
            state, _ = video_env.reset()
            terminated, truncated = False, False
            while not (terminated or truncated):
                action = ppo_agent.select_action(state)
                state, _, terminated, truncated, _ = video_env.step(action)

            print(f"Video saved for episode {episode}")

        video_env.close()
        print(f"Videos saved to {video_dir}")
        writer.close()

    # Testing function
    def test_agent(env, agent):
        test_agent_policy = agent
   
        rewards = []
        successes = 0

        for i in range(args.n_test_episodes):
            state, _ = env.reset()
            cumulative_reward = 0
            terminated = False
            truncated = False

            if args.use_attention:
                constraints = dummy_env.encode_constraints()
                constraints = torch.tensor(constraints, dtype=torch.float32).to(device)
            else:
                constraints = None

            for j in range(args.max_timesteps):
                action = test_agent_policy.select_action(state, constraints=constraints, testing=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                cumulative_reward += reward
                state = next_state

                if terminated or truncated:
                    break

            rewards.append(cumulative_reward)
            if args.use_smallenv:
                if "iron_sword" in env.inventory:
                    successes += 1
            else:
                if "treasure" in env.inventory:
                    successes += 1

        success_rate = successes / args.n_test_episodes
        return rewards, success_rate

    # Start training
    train()
    return log_dir, args

if __name__ == "__main__":
    main()