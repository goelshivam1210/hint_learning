from datetime import datetime
import os
import yaml  # Import the YAML library
import torch
import numpy as np
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

# Import your environment and wrapper
from env import SimpleEnv
from env_wrapper import EnvWrapper

# Hyperparameters
lr = 1e-5
gamma = 0.99
epoch = 1000
batch_size = 64
n_step = 5
step_per_epoch = 2000
step_per_collect = 1000
repeat_per_collect = 4
episode_per_test = 20
buffer_size = 20000

# Success rate threshold
convergence_threshold = 0.9  # 90%
convergence_window = 10  # Last 10 evaluations

# Device setup
device = torch.device("cpu") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create environment and wrap with constraints
def make_env():
    env = SimpleEnv(render_mode=None)  # Create the base environment
    wrapped_env = EnvWrapper(env, "constraints.yaml")  # Wrap with constraint handler
    return wrapped_env

train_envs = DummyVectorEnv([make_env for _ in range(8)])  # 8 parallel environments for training
test_envs = DummyVectorEnv([make_env for _ in range(8)])   # 8 parallel environments for testing

# Observation and action space
dummy_env = make_env()
action_space = dummy_env.action_space
obs_space = dummy_env.observation_space
# print (f"observation space = ", obs_space)

combined_shape = obs_space.shape

# Define the Actor and Critic networks for PPO
net = Net(combined_shape, hidden_sizes=[256, 64], device=device)
actor = Actor(net, action_space.n, device=device).to(device)
critic = Critic(net, device=device).to(device)
actor_critic = ActorCritic(actor, critic).to(device)

# Optimizer
optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)

# PPO Policy with Categorical distribution for discrete action spaces
ppo_policy = PPOPolicy(
    actor=actor,
    critic=critic,
    optim=optim,
    dist_fn=torch.distributions.Categorical,  # Use categorical distribution for discrete actions
    discount_factor=gamma,
    gae_lambda=0.95,
    max_grad_norm=0.5,
    vf_coef=0.5,
    ent_coef=0.01,
    eps_clip=0.2,
    reward_normalization=False,
    action_scaling=False,
    action_space=action_space
).to(device)

# Generate a unique directory name based on the current time
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join('log', f'ppo_{timestamp}')

# Set up Tensorboard Logger
writer = SummaryWriter(log_dir)
logger = TensorboardLogger(writer)

# Collectors
train_collector = Collector(ppo_policy, train_envs, VectorReplayBuffer(buffer_size, buffer_num=8))
test_collector = Collector(ppo_policy, test_envs)

# Define a custom stopping function based on the success rate
success_history = []

def stop_fn(mean_rewards):
    global success_history
    
    # Collect the terminated flags from all episodes in the test collector's buffer
    terminated_flags = [info['terminated'] for info in test_collector.buffer.info if 'terminated' in info]
    
    # Count successes based on terminated flags (terminated == True indicates success)
    successes = np.sum(terminated_flags)

    # Update success history
    success_history.append(successes)

    # Limit the success history to the last `convergence_window` evaluations
    if len(success_history) > convergence_window:
        success_history = success_history[-convergence_window:]

    # Calculate the success rate over the last convergence_window evaluations
    success_rate = np.mean(success_history) / episode_per_test  # Normalize by the number of test episodes

    print(f"Success Rate: {success_rate * 100:.2f}%")

    # If the success rate exceeds the convergence threshold, stop training
    return success_rate >= convergence_threshold

# Trainer
trainer = OnpolicyTrainer(
    policy=ppo_policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=epoch,
    step_per_epoch=step_per_epoch,
    step_per_collect=step_per_collect,
    episode_per_test=episode_per_test,
    batch_size=batch_size,
    repeat_per_collect=repeat_per_collect,
    stop_fn=stop_fn,  # Add the stopping function
    logger=logger
)

# Run the trainer with success tracking
for epoch_idx in range(epoch):
    result = trainer.run()

    # Calculate the number of successes after each epoch using the `terminated` flag
    buffer_size = len(test_collector.buffer)
    if buffer_size > 0:
        batch = test_collector.buffer.sample(buffer_size)[0]
        terminated_flags = batch.info.get("terminated", np.zeros_like(batch.rew))
        successes = np.sum(terminated_flags)
        print(f"Epoch #{epoch_idx + 1}: Number of successes: {successes} / {episode_per_test}")

    # Check for convergence and stop if needed
    if stop_fn(result.best_reward):
        print(f"Converged at epoch {epoch_idx + 1}. Stopping training.")
        break

# Print the final results of training
print(f"Training finished! Result: {result}\n")
print(f"Best reward: {result.best_reward}\n")
print(f"Best reward standard deviation: {result.best_reward_std}\n")
print(f"Average reward: {result.best_score}\n")
print(f"Train steps: {result.train_step}\n")
print(f"Train episodes: {result.train_episode}\n")
print(f"Test steps: {result.test_step}\n")
print(f"Test episodes: {result.test_episode}\n")
print(f"Total training time: {result.timing.total_time:.2f} seconds")

# Close the writer
writer.close()