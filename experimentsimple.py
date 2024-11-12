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

models_steps_test = []
models_average_values_test = []
models_standard_devs_test = []

models_steps_train = []
models_average_values_train = []
models_standard_devs_train = []


values_test = {}
values_train = {}


# event_file1 = "shivam_logs_2/log/ppo_instance_20241030_013156_hints_False_attention_False/logs/events.out.tfevents.1730266316.Mac.88299.0"

# # event_file2 = "shivam_logs_2/log/ppo_instance_20241030_013249_hints_True_attention_False/logs/events.out.tfevents.1730266369.Mac.88613.0"

# event_file3 = "shivam_logs_2/log/ppo_instance_20241030_013212_hints_False_attention_False/logs/events.out.tfevents.1730266332.Mac.88511.0"

# event_file4 = "shivam_logs_2/log/ppo_instance_20241030_013219_hints_False_attention_False/logs/events.out.tfevents.1730266339.Mac.88526.0"

# event_file5 = "shivam_logs_2/log/ppo_instance_20241030_013227_hints_False_attention_False/logs/events.out.tfevents.1730266347.Mac.88561.0"

# event_files = [event_file1, event_file3, event_file4, event_file5]

for trial in range(num_trials):
    print(f"TRIAL {trial}")
    log_dir = run.main()
    event_file = os.path.join(log_dir, os.listdir(log_dir)[0])

    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    testScalars = event_acc.Scalars('Test/Success_Rate')
    trainScalars = event_acc.Scalars("Train/Success_Rate")

    for scalar in testScalars:
        step = scalar.step
        value = scalar.value
        
        # Accumulate value for each episode
        if step not in values_test:
            values_test[step] = []

        values_test[step].append(value)
    
    for scalar in trainScalars:
        step = scalar.step
        value = scalar.value
        
        # Accumulate value for each episode
        if step not in values_train:
            values_train[step] = []

        values_train[step].append(value)

steps_test = sorted(values_test.keys())
average_values_test = [np.mean(values_test[step]) for step in steps_test]
standard_devs_test = [np.std(values_test[step]) for step in steps_test]
models_steps_test.append(steps_test)
models_average_values_test.append(average_values_test)
models_standard_devs_test.append(standard_devs_test)

steps_train = sorted(values_train.keys())
average_values_train = [np.mean(values_train[step]) for step in steps_train]
standard_devs_train = [np.std(values_train[step]) for step in steps_train]
models_steps_train.append(steps_train)
models_average_values_train.append(average_values_train)
models_standard_devs_train.append(standard_devs_train)


plt.plot(models_steps_test[0], models_average_values_test[0], label='Base', color = 'b')
plt.fill_between(models_steps_test[0], np.array(models_average_values_test[0]) - np.array(models_standard_devs_test[0]), 
                                  np.array(models_average_values_test[0]) + np.array(models_standard_devs_test[0]), 
                                  color='b', alpha=0.2)


plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.title('Testing Curve')

plt.legend()

# Save the plot
save_path = os.path.join("shivam_logs_2/log/", instance_id + "test")
plt.savefig(save_path)
plt.show()

plt.clf()

plt.plot(models_steps_train[0], models_average_values_train[0], label='Base', color = 'b')
plt.fill_between(models_steps_train[0], np.array(models_average_values_train[0]) - np.array(models_standard_devs_train[0]), 
                                  np.array(models_average_values_train[0]) + np.array(models_standard_devs_train[0]), 
                                  color='b', alpha=0.2)


plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.title('Training Curve')
plt.legend()

# Save the plot
save_path = os.path.join("shivam_logs_2/log/", instance_id + "train")
plt.savefig(save_path)
plt.show()