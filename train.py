import torch
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
from tianshou.utils.net.common import Net

from env import SimpleEnv

from gymnasium.spaces import Box, Discrete, MultiBinary, MultiDiscrete



# Hyperparameters
lr, epoch, batch_size = 1e-3, 10, 64
gamma, n_step, target_freq = 0.9, 3, 320
buffer_size = 20000
eps_train, eps_test = 0.1, 0.05
step_per_epoch, step_per_collect = 10000, 10

# Logger setup
logger = ts.utils.TensorboardLogger(SummaryWriter('log/dqn'))

# Check if MPS is available (on macOS)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Create environment wrappers for Tianshou
def make_env():
    return SimpleEnv(render_mode=None)

# Step 2: Create vectorized environments for parallel processing
train_envs = ts.env.DummyVectorEnv([make_env for _ in range(8)])  # 8 parallel environments for training
test_envs = ts.env.DummyVectorEnv([make_env for _ in range(8)])   # 8 parallel environments for testing

# Step 3: Ensure the state and action shapes are correctly retrieved
dummy_env = make_env()  # Create a single instance of the environment for shape checking
obs_space = dummy_env.observation_space  # Fetch observation space from a single env instance
action_space = dummy_env.action_space  # Fetch action space from a single env instance
# print (f"Action space type = {type(action_space)}")
# print (f"Observation space = {type(obs_space)}")
# print (f"Action space = {action_space}")



lidar_shape = obs_space['lidar'].shape
inventory_shape = obs_space['inventory'].shape

    # Calculate the combined state shape by summing the dimensions of lidar and inventory
state_shape = lidar_shape
# Handle discrete and continuous action spaces
action_shape = action_space.n  # For Discrete, treat it as a single value


print("State shape: ", state_shape)
print("Action shape: ", action_shape)

# Define the network
net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128], device=device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

print(f"Type of action space: {isinstance(action_space, Discrete)} ")
print (f"action space = {action_space}")

# Policy setup (DQN)
policy = ts.policy.DQNPolicy(
    model=net,
    optim=optim,
    discount_factor=gamma, 
    estimation_step=n_step,
    target_update_freq=target_freq,
    action_space=action_space
)

# Training and Testing Collectors
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, len(train_envs)), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method

# Training process
result = ts.trainer.OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=epoch,
    step_per_epoch=step_per_epoch,
    step_per_collect=step_per_collect,
    episode_per_test=10,
    batch_size=batch_size,
    update_per_step=1 / step_per_collect,
    train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
    test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
    stop_fn=lambda mean_rewards: mean_rewards >= 100,  # Use a reasonable threshold
    logger=logger,
).run()

print(f"Finished training in {result['duration']} seconds")

# Save and load the policy
torch.save(policy.state_dict(), 'dqn.pth')
policy.load_state_dict(torch.load('dqn.pth'))

# Evaluation
policy.eval()
policy.set_eps(eps_test)
collector = ts.data.Collector(policy, dummy_env, exploration_noise=True)
collector.collect(n_episode=1, render=1 / 35)