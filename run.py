import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import argparse
from ppo import PPO


# Import environment and wrapper
from env import SimpleEnv
from env_wrapper import EnvWrapper

# Import attention network
from attention_net_own import AttentionNet

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO with optional attention and environment wrapping.')

    # Environment and attention flags
    parser.add_argument('--use-wrapper', action='store_true', help='Use environment wrapper with constraint encoding.')
    parser.add_argument('--use-attention', action='store_true', help='Use attention mechanism in PPO.')
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'], help='Device to run the model on.')

    # Hyperparameters
    parser.add_argument('--lr-actor', type=float, default=0.0003, help='Learning rate for the actor.')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='Learning rate for the critic.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for PPO.')
    parser.add_argument('--K-epochs', type=int, default=80, help='Number of PPO epochs per update.')
    parser.add_argument('--eps-clip', type=float, default=0.2, help='Clip range for PPO updates.')
    parser.add_argument('--max-episodes', type=int, default=100000, help='Maximum number of training episodes.')
    parser.add_argument('--max-timesteps', type=int, default=300, help='Maximum number of timesteps per episode.')
    parser.add_argument('--update-timestep', type=int, default=2000, help='Timesteps after which PPO update is triggered.')
    parser.add_argument('--save-interval', type=int, default=1000, help='Interval to save the model.')
    parser.add_argument('--log-interval', type=int, default=10, help='Interval to log training progress.')
    parser.add_argument('--test-interval', type=int, default=20, help='Interval to test the agent.')
    parser.add_argument('--n-test-episodes', type=int, default=20, help='Number of test episodes.')

    args = parser.parse_args()
    return args


# Utility function to flatten dict observation space for the unwrapped environment
def flatten_obs_space(obs_dict):
    if isinstance(obs_dict, dict) and 'lidar' in obs_dict and 'inventory' in obs_dict:
        lidar_obs = obs_dict['lidar'].flatten()
        inventory_obs = obs_dict['inventory']
        return np.concatenate([lidar_obs, inventory_obs], axis=-1)
    else:
        # If it's already flattened, return it as is
        return obs_dict


# Main function
def main():
    args = parse_args()

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

    # Environment setup
    def make_env():
        env = SimpleEnv(render_mode=None)  # Create the base environment
        if args.use_wrapper:
            wrapped_env = EnvWrapper(env, "constraints.yaml")  # Wrap with constraint handler
            return wrapped_env
        else:
            return env  # Return the base environment without wrapping

    env = make_env()
    test_env = make_env()
    dummy_env = make_env()
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

    # Training Loop
    def train():
        timestep = 0
        success_train = 0
        for episode in range(1, args.max_episodes + 1):
            state, _ = env.reset()

            cumulative_reward = 0

            # Encode constraints (if using attention)
            constraints = dummy_env.encode_constraints() if args.use_attention else None
            if args.use_attention:
                constraints = torch.tensor(constraints, dtype=torch.float32).to(device)  # Convert to tensor

            for t in range(args.max_timesteps):
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
                if timestep % args.update_timestep == 0:
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
                writer.add_scalar("Test/Reward", np.mean(test_rewards), episode)
                writer.add_scalar("Test/Success_Rate", success_rate, episode)
                print(f"Test Results - Episode {episode}: Avg Reward: {np.mean(test_rewards):.2f}, Success Rate: {success_rate * 100:.2f}%")

        writer.close()

    # Testing function
    def test_agent(env, agent):
        test_agent_policy = PPO(
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

        test_agent_policy.policy.load_state_dict(agent.policy.state_dict())
        rewards = []
        successes = 0

        for _ in range(args.n_test_episodes):
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
            if "treasure" in env.inventory:
                successes += 1

        success_rate = successes / args.n_test_episodes
        return rewards, success_rate

    # Start training
    train()

if __name__ == "__main__":
    main()