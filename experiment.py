import os
import torch
import numpy as np
import run
from datetime import datetime
import argparse
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter
from ppo import PPO
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Create a unique identifier for this plot
num_trials = 10
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
instance_id = f"plot_{timestamp}_next_{num_trials}_models"

# Dictionary to store cumulative values and counts for averaging
cumulative_values = {}
counts = {}

for i in range(num_trials):
    log_dir = run.main()
    event_file = os.path.join(log_dir, os.listdir(log_dir)[0])

    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    scalars = event_acc.Scalars('Test/Success_Rate')

    for scalar in scalars:
        step = scalar.step
        value = scalar.value
        
        # Accumulate value for each episode
        if step not in cumulative_values:
            cumulative_values[step] = 0.0
            counts[step] = 0

        cumulative_values[step] += value
        counts[step] += 1  # Track how many trials have provided data for this episode


average_steps = sorted(cumulative_values.keys())
average_values = [cumulative_values[step] / counts[step] for step in average_steps]


plt.plot(average_steps, average_values, label='Test/Success Rate')
plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.legend()

# Save the plot
save_path = os.path.join("log", instance_id)
plt.savefig(save_path)

# Show the plot
plt.show()