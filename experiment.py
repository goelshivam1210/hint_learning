import os
import torch
import numpy as np
import run
from datetime import datetime
import argparse
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter
from ppo import PPO
import sys
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Create a unique identifier for this plot
num_trials = 5
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
instance_id = f"plot_{timestamp}_next_{num_trials}_models"

wrapperArg = "--use-wrapper"
attentionArg = "--use-attention"

models_steps = []
models_average_values = []
models_standard_devs = []


for i in range(3):
    values = {}
    if i == 1:
        sys.argv.append(wrapperArg)
    if i == 2:
        sys.argv.append(attentionArg)
    print(sys.argv)
    for trial in range(num_trials):
        print(f"TRIAL {trial}")
        log_dir = run.main()
        event_file = os.path.join(log_dir, os.listdir(log_dir)[0])

        event_acc = EventAccumulator(event_file)
        event_acc.Reload()

        scalars = event_acc.Scalars('Test/Success_Rate')

        for scalar in scalars:
            step = scalar.step
            value = scalar.value
            
            # Accumulate value for each episode
            if step not in values:
                values[step] = []

            values[step].append(value)

    steps = sorted(values.keys())
    average_values = [np.mean(values[step]) for step in steps]
    standard_devs = [np.std(values[step]) for step in steps]
    models_steps.append(steps)
    models_average_values.append(average_values)
    models_standard_devs.append(standard_devs)



plt.plot(models_steps[0], models_average_values[0], label='Base', color = 'b')
plt.fill_between(models_steps[0], np.array(models_average_values[0]) - np.array(models_standard_devs[0]), 
                                  np.array(models_average_values[0]) + np.array(models_standard_devs[0]), 
                                  color='b', alpha=0.2)


plt.plot(models_steps[1], models_average_values[1], label='UW=true, UA=false', color='r')
plt.fill_between(models_steps[1], np.array(models_average_values[1]) - np.array(models_standard_devs[1]), 
                                  np.array(models_average_values[1]) + np.array(models_standard_devs[1]), 
                                  color='r', alpha=0.2)

plt.plot(models_steps[2], models_average_values[2], label='UW=true, UA=true', color='g')
plt.fill_between(models_steps[2], np.array(models_average_values[2]) - np.array(models_standard_devs[2]), 
                                  np.array(models_average_values[2]) + np.array(models_standard_devs[2]), 
                                  color='g', alpha=0.2)


plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.legend()

# Save the plot
save_path = os.path.join("log", instance_id)
plt.savefig(save_path)

# Show the plot
plt.show()