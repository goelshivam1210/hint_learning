import os
import random
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter
from ppo import PPO
import yaml

# Import environment and wrapper
from editedenv import SimpleEnv
from env2 import SimpleEnv2, RewardType

from env_wrapper import EnvWrapper
# from env_wrapper_2 import EnvWrapper

from data_collector import TrajectoryProcessor, TransitionGraph

# Import attention network
from attention_net_own import AttentionNet

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO with optional attention and environment wrapping.')
    # Environment and attention flags
    parser.add_argument('--use_wrapper', action='store_true', help='Use environment wrapper with constraint encoding.')
    parser.add_argument('--use_attention', action='store_true', help='Use attention mechanism in PPO.')
    parser.add_argument('--use_smallenv', action='store_true', help='Use environment with smaller action space')
    parser.add_argument('--use_dense', action='store_true', help='Use dense reward function')
    parser.add_argument('--use_graph_reward', action='store_true', help='Enable graph-based reward shaping.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for logging.')

    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'], help='Device to run the model on.')
    parser.add_argument('--load-model', type=str, default=None, help='Path to a saved PPO model to resume training.')
    parser.add_argument('--logdir', type=str, default=None, help='Path to an existing TensorBoard log directory to resume training.')

    # Hyperparameters
    parser.add_argument('--lr-actor', type=float, default=0.0003, help='Learning rate for the actor.')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='Learning rate for the critic.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for PPO.')
    parser.add_argument('--K-epochs', type=int, default=5, help='Number of PPO epochs per update.')
    parser.add_argument('--eps-clip', type=float, default=0.2, help='Clip range for PPO updates.')
    parser.add_argument('--grid_size', type=int, default=12, help='Size of the gridworld')
    parser.add_argument('--max_episodes', type=int, default=20000, help='Maximum number of training episodes.')
    parser.add_argument('--max_timesteps', type=int, default=500, help='Maximum number of timesteps per episode.')
    parser.add_argument('--update_timestep', type=int, default=4000, help='Timesteps after which PPO update is triggered.')
    parser.add_argument('--batch_size', type=int, default=128, help="how many collected timesteps (from the environment rollouts) are used in one gradient update.")
    parser.add_argument('--save_interval', type=int, default=1000, help='Interval to save the model.')
    parser.add_argument('--log_interval', type=int, default=500, help='Interval to log training progress.')
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
    # instance_id = f"ppo_instance_{timestamp}_hints_{args.use_wrapper}_attention_{args.use_attention}"
    instance_id = f"ppo_instance_{timestamp}_hints_{args.use_wrapper}_attention_{args.use_attention}_graph_{args.use_graph_reward}"
    # Create a directory for saving models and logs for this training instance
    base_dir = os.path.join("log", instance_id)
    os.makedirs(base_dir, exist_ok=True)

    # Model save directory
    model_save_dir = os.path.join(base_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)

    # TensorBoard setup
    log_dir = args.logdir if args.logdir else os.path.join(base_dir, "logs")
    writer = SummaryWriter(log_dir)

    # # Sample data (Python dictionary)
    # param_dict = vars(args)

    # with open(os.path.join(base_dir, "params.yaml"), 'w') as file:
    #     yaml.dump(param_dict, file)

    # Environment setup
    def make_env(seed=args.seed):
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
        
        if args.seed is not None:
            env.reset(seed=args.seed)

        if args.use_wrapper:
            wrapped_env = EnvWrapper(env, "constraints.yaml")
            # wrapped_env = EnvWrapper(env)
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
            # wrapped_env = EnvWrapper(env)
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

    if args.load_model:
        print(f"Loading model from {args.load_model}")
        ppo_agent.load(args.load_model)

        try:
            last_episode = int(args.load_model.split("_")[-1].split(".")[0])  # Extract episode number from filename
        except ValueError:
            last_episode = 0  # If parsing fails, start fresh
    else:
        last_episode = 0  # If no model is loaded, start fresh


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

    # save logs for debugging
    log_file = os.path.join(base_dir, "debug_log.txt")
    with open(log_file, "w") as f:
        f.write("=== Debug Log ===\n")  # Create log header

    # Training Loop
    def train():
        processor = TrajectoryProcessor(constraint_file="constraints.yaml",
                                        graph_constraints=[
                                        "inventory(iron_sword) > 0",
                                        "facing(iron_ore)",
                                        "inventory(iron) > 0",      
                        ])
        transition_graph = TransitionGraph()
        timesteps_per_episode = []
        timestep = 0
        success_train = 0
        success_window = []
        converged = False
        
        # log debug 
        log_file = os.path.join(base_dir, "debug_log.txt")
        if args.debug:
            with open(log_file, "w") as f:
                f.write("=== Debug Log ===\n")
        
        for episode in range(last_episode + 1, args.max_episodes + 1):
            state, _ = env.reset()
            state = (state - state.mean()) / state.std()
            trajectory = []  # Store the trajectory
            cumulative_reward = 0
            cumulative_graph_reward = 0
  
            if args.debug:
                with open(log_file, "a") as f:
                    f.write(f"\n[DEBUG] Episode {episode} START\n")

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

                # Apply graph-based reward shaping if enabled
                if args.use_graph_reward:
                    prev_state_graph = processor.extract_constraints(state)  
                    new_state_graph = processor.extract_constraints(next_state)
                    transition = (tuple(sorted(prev_state_graph.items())), tuple(sorted(new_state_graph.items())))
           
                    # Compute graph-based reward
                    graph_reward = (
                        5 * transition_graph.compute_reward().get(transition, 0)
                        if len(transition_graph.graph.edges) > 0 else 0
                    )
                    reward += graph_reward
                    cumulative_graph_reward += graph_reward  # Accumulate graph-based reward per episode

                    if args.debug:
                        if cumulative_graph_reward > 0:
                            with open(log_file, "a") as f:
                                f.write(f"[DEBUG] Step {t_step}: Graph Reward={graph_reward}, Cumulative={cumulative_graph_reward}\n")

                # Store transition for graph-based reward shaping
                trajectory.append({"state": state, "action": action, "reward": reward})

                next_state = (next_state - next_state.mean()) / next_state.std()

                # Store transition
                trajectory.append({"state": state, "action": action, "reward": reward})
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(terminated or truncated)

                # Normalize next state and move forward
                state = (next_state - next_state.mean()) / next_state.std()
                cumulative_reward += reward

                    # Stop episode if terminated or truncated
                if terminated or truncated:
                    timesteps_per_episode.append(t_step + 1)

                    # If goal state was reached, process trajectory
                    if env.treasure_obtained:
                        success_train += 1

                        if args.debug:
                            with open(log_file, "a") as f:
                                f.write(f"[DEBUG] Episode {episode}: Goal Reached!\n")

                        if args.use_graph_reward:
                            processed_trajectory = [processor.extract_constraints(t["state"]) for t in trajectory]
                            processor.store_trajectory(processed_trajectory)
                            transition_graph.add_trajectory(processed_trajectory)

                    break  # Stop processing further steps

            # Periodically update transition graph
            if episode % 500 == 0 and args.use_graph_reward:
                processed_trajectory = [processor.extract_constraints(t["state"]) for t in trajectory]
                processor.store_trajectory(processed_trajectory)
                transition_graph.add_trajectory(processed_trajectory)
                transition_graph.visualize_graph(save_path=os.path.join(base_dir, f"graph_{episode}.png"))

                if args.debug:
                    with open(log_file, "a") as f:
                        f.write(f"[DEBUG] Episode {episode}: Transition Graph Updated\n")

            # PPO update step
            if timestep % args.update_timestep == 0:
                ppo_agent.update()
                if args.debug:
                    with open(log_file, "a") as f:
                        f.write(f"[DEBUG] PPO Update at timestep {timestep}\n")

            # Logging
            writer.add_scalar("Train/Reward", cumulative_reward, timestep)
            writer.add_scalar("Train/Success_Rate", success_train / episode, timestep)
            writer.add_scalar("Train/Timesteps_Per_Episode", np.mean(timesteps_per_episode), timestep)
            writer.add_scalar("Train/Graph_Reward", cumulative_graph_reward, timestep)

            # Periodic logging and model saving
            if episode % args.log_interval == 0:
                print(f"Episode {episode}/{args.max_episodes}, Reward: {cumulative_reward:.2f}")

            ppo_agent.save(os.path.join(model_save_dir, "ppo_latest.pth"))
            if episode % args.save_interval == 0 or episode == args.max_episodes - 1:
                model_path = os.path.join(model_save_dir, f"ppo_model_{episode}.pth")
                ppo_agent.save(model_path)
                print(f"Checkpoint saved at episode {episode} to {model_path}")

            # Periodic testing
            if episode % args.test_interval == 0:
                test_rewards, success_rate = test_agent(test_env, ppo_agent)
                success_window.append(success_rate)
                success_window = success_window[-args.convergence:]  # Keep only last `convergence` tests

                if args.debug:
                    with open(log_file, "a") as f:
                        f.write(f"[DEBUG] Test Success Rate: {success_rate:.2f}\n")

                # Check for convergence
                if len(success_window) == args.convergence and np.mean(success_window) >= 0.95:
                    converged, avg_success, improvement = has_converged(success_window, args.epsilon)
                    if converged:
                        print(f"Converged with success rate: {avg_success:.2f} (Improvement: {improvement:.2f})")
                        ppo_agent.save(os.path.join(model_save_dir, f"ppo_model_converged_{episode}.pth"))
                        print(f"Converged and saved model at {model_path}")
                        break

        # Save training videos
        video_dir = os.path.join(base_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

        video_env = RecordVideo(make_env_for_video(), video_dir, episode_trigger=lambda x: x < 5)
        for episode in range(1, 6):
            state, _ = video_env.reset()
            while not any(video_env.step(ppo_agent.select_action(state))[2:4]):  # Run until terminated/truncated
                pass
            print(f"Video saved for episode {episode}")

        video_env.close()
        print(f"Videos saved to {video_dir}")
        writer.close()           
                    
        #         # If a successful trajectory is found, process it
        #         if "treasure" in env.inventory:
        #             print ("Training: Goal state reached!")
        #             # print(f"[DEBUG] Episode {episode}: Goal reached! Storing trajectory.")

        #             success_train += 1
                    
        #             # Process and store trajectory -- graph based reward shaping
        #             if args.use_graph_reward:
        #                 processed_trajectory = [processor.extract_constraints(t["state"]) for t in trajectory]
        #                 # print(f"[DEBUG] Processed Trajectory States: {processed_trajectory}")
        #                 processor.store_trajectory(processed_trajectory)
        #                 # Add trajectory to the transition graph
        #                 transition_graph.add_trajectory(processed_trajectory)

        #                 # with open(log_file, "a") as f:
        #                 #     f.write(f"[DEBUG] Episode {episode}: Added Edges -> {list(transition_graph.graph.edges)}\n")
        #                 #     f.write(f"[DEBUG] Episode {episode}: Before Pruning, Graph Edges -> {list(transition_graph.graph.edges)}\n")
        #                 # Optimize graph by removing redundant edges
                        
        #                 # transition_graph.prune_graph()
                        
        #                 # with open(log_file, "a") as f:
        #                 #     f.write(f"[DEBUG] Episode {episode}: Nodes AFTER Pruning -> {list(transition_graph.graph.nodes)}\n")
        #                 #     f.write(f"[DEBUG] Episode {episode}: Edges AFTER Pruning -> {list(transition_graph.graph.edges)}\n")

        #                 # save the graph
        #                 # graph_save_path = os.path.join(base_dir, f"graph_{episode}.png")
        #                 # transition_graph.visualize_graph(save_path=graph_save_path)

        #         # Periodically update the transition graph
        #         if episode % 500 == 0 and args.use_graph_reward is True:
        #             processed_trajectory = [processor.extract_constraints(t["state"]) for t in trajectory]
        #             processor.store_trajectory(processed_trajectory)
        #             transition_graph.add_trajectory(processed_trajectory)
        #             # print(f"Episode {episode}: Updating transition graph...")
        #             # transition_graph.prune_graph()
        #             # transition_graph.visualize_graph()
        #             graph_save_path = os.path.join(base_dir, f"graph_{episode}.png")
        #             transition_graph.visualize_graph(save_path=graph_save_path)


        #         if timestep % args.update_timestep == 0:
        #             ppo_agent.update()

        #         if terminated:
        #             timsteps_per_episode.append(t_step+1)
        #             break
        #         if truncated:
        #             timsteps_per_episode.append(t_step+1)
        #             break
            
        #     # with open(log_file, "a") as f:
        #         # f.write(f"[DEBUG] Episode {episode} END: Final Cumulative Graph Reward={cumulative_graph_reward}\n")
        #     # TensorBoard logging
        #     writer.add_scalar("Train/Reward", cumulative_reward, timestep)
        #     writer.add_scalar("Train/Success_Rate", success_train / episode, timestep)
        #     writer.add_scalar("Train/Timesteps_Per_Episode", np.mean(timsteps_per_episode), timestep)
        #     writer.add_scalar("Train/Graph_Reward", cumulative_graph_reward, timestep)  # Log graph-based reward

        #     # Print and log
        #     if episode % args.log_interval == 0:
        #         print(f"Episode {episode}/{args.max_episodes}, Reward: {cumulative_reward:.2f}")

        #     # Save latest model (overwrites every time)
        #     latest_model_path = os.path.join(model_save_dir, "ppo_latest.pth")
        #     ppo_agent.save(latest_model_path)
        #     # print(f"Latest model saved at {latest_model_path}")

        #     # Save periodic checkpoints
        #     if episode % args.save_interval == 0 or episode == args.max_episodes-1:
        #         model_path = os.path.join(model_save_dir, f"ppo_model_{episode}.pth")
        #         ppo_agent.save(model_path)
        #         print(f"Checkpoint saved at episode {episode} to {model_path}")

        #         # Save graph if enabled
        #         if args.use_graph_reward:
        #             graph_save_path = os.path.join(base_dir, f"graph_{episode}.png")
        #             transition_graph.visualize_graph(save_path=graph_save_path)
        #             # plt.savefig(graph_save_path)
        #             # print(f"Graph saved at: {graph_save_path}")

        #     # Test the agent periodically
        #     if episode % args.test_interval == 0:
        #         test_rewards, success_rate = test_agent(test_env, ppo_agent)
        #         # Append the success rate to the sliding window
        #         success_window.append(success_rate)

        #         # Maintain the window size (last 10 test results)
        #         if len(success_window) > args.convergence:
        #             success_window.pop(0)

        #         # Check for convergence: if the average success rate over the last 10 tests is > 95%
        #         if len(success_window) == args.convergence and np.mean(success_window) >= 0.95:
        #             converged, avg_success, improvement = has_converged(success_window, args.epsilon)
        #             if converged:
        #                 print(f"Converged with success rate: {avg_success:.2f} (Improvement: {improvement:.2f})")

        #         # Log test metrics using total timesteps
        #         writer.add_scalar("Test/Reward", np.mean(test_rewards), timestep)
        #         writer.add_scalar("Test/Success_Rate", success_rate, timestep)
        #         print(f"Test Episode {episode}: Avg Reward: {np.mean(test_rewards):.2f}, Success Rate: {success_rate * 100:.2f}%")

        #         # Stop training if converged
        #         if converged:
        #             model_path = os.path.join(model_save_dir, f"ppo_model_converged_{episode}.pth")
        #             ppo_agent.save(model_path)
        #             print(f"Converged and saved model at {model_path}")
        #             break
        # # Directory to save the video
        # video_dir = os.path.join(base_dir, "videos")
        # os.makedirs(video_dir, exist_ok=True)

        # # Set the episode trigger to only save video for the first 5 episodes
        # video_env = RecordVideo(make_env_for_video(), video_dir, episode_trigger=lambda x: x < 5)  # Save videos for episodes 1-5

        # # Loop through episodes for video recording (limited to the first 5)
        # for episode in range(1, 6):  # Loop only for 5 episodes
        #     state, _ = video_env.reset()
        #     terminated, truncated = False, False
        #     while not (terminated or truncated):
        #         action = ppo_agent.select_action(state)
        #         state, _, terminated, truncated, _ = video_env.step(action)

        #     print(f"Video saved for episode {episode}")

        # video_env.close()
        # print(f"Videos saved to {video_dir}")
        # writer.close()

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
                    if env.treasure_obtained:
                        successes += 1
                    break

            rewards.append(cumulative_reward)

        success_rate = successes / args.n_test_episodes
        return rewards, success_rate
    
    def has_converged(success_window, epsilon, threshold=0.95):
        """
        Check if the agent has converged based on success rates.
        
        Args:
            success_window (list): Sliding window of recent success rates.
            epsilon (float): Minimum improvement over the window to avoid convergence.
            threshold (float): Success rate threshold for convergence.

        Returns:
            bool: Whether convergence criteria are met.
            float: Average success rate over the window.
            float: Improvement over the window.
        """
        avg_success = np.mean(success_window)
        improvement = success_window[-1] - success_window[0]
        return avg_success >= threshold and improvement <= epsilon, avg_success, improvement

    # Start training
    train()
    return log_dir, args

if __name__ == "__main__":
    main()