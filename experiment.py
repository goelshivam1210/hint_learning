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
num_trials = 2
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
instance_id = f"plot_{timestamp}_next_{num_trials}_models"

wrapperArg = "--use-wrapper"
attentionArg = "--use-attention"

models_average_values_test = []
models_standard_devs_test = []

models_average_values_train = []
models_standard_devs_train = []

for i in range(3):
    trial_results_test = []
    trial_results_train = []

    if i == 1:
        sys.argv.append(wrapperArg)
    if i == 2:
        sys.argv.append(attentionArg)
    print(sys.argv)
    for trial in range(num_trials):
        print(f"TRIAL {trial}")
        log_dir, args = run.main()
        event_file = os.path.join(log_dir, os.listdir(log_dir)[0])

        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        test_results = []  # Stores values for this trial
        train_results = []  # Stores values for this trial

        testScalars = event_acc.Scalars('Test/Success_Rate')
        trainScalars = event_acc.Scalars("Train/Success_Rate")

        for scalar in testScalars:
            test_results.append(scalar.value)
        
        for scalar in trainScalars:
            train_results.append(scalar.value)

        trial_results_test.append(test_results)
        trial_results_train.append(train_results)




    def align_trials(trials):
        max_length = max(len(trial) for trial in trials)
        aligned_trials = []
        for trial in trials:
            # Extend each trial with its last value
            if len(trial) < max_length:
                trial.extend([trial[-1]] * (max_length - len(trial)))
            aligned_trials.append(trial)
        return aligned_trials

    aligned_test_results = align_trials(trial_results_test)
    aligned_train_results = align_trials(trial_results_train)

    average_values_test = np.mean(aligned_test_results, axis=0)
    standard_devs_test = np.std(aligned_test_results, axis=0)
    average_values_train = np.mean(aligned_train_results, axis=0)
    standard_devs_train = np.std(aligned_train_results, axis=0)

    models_average_values_test.append(average_values_test)
    models_standard_devs_test.append(standard_devs_test)
    models_average_values_train.append(average_values_train)
    models_standard_devs_train.append(standard_devs_train)


env_string = "small" if args.use_smallenv else "normal"
test_plot_title = f"Test Success Rate (env : {env_string}) , Trials: {num_trials}"
train_plot_title = f"Train Success Rate (env : {env_string}), Trials: {num_trials}"

# Plotting results for test success rate
plt.figure(figsize=(10, 6))
colors = ['b', 'r', 'g']
labels = ['Base', 'UW=true, UA=false', 'UW=true, UA=true']

for i in range(3):
    plt.plot(range(len(models_average_values_test[i])),
             models_average_values_test[i],
             label=labels[i], color=colors[i])
    plt.fill_between(range(len(models_average_values_test[i])),
                     models_average_values_test[i] - models_standard_devs_test[i],
                     models_average_values_test[i] + models_standard_devs_test[i],
                     color=colors[i], alpha=0.2)

plt.title(test_plot_title)
plt.xlabel('Episode')
plt.ylabel('Test Success Rate')
plt.legend()
save_path_test = os.path.join("log", f"{instance_id}_test")
plt.savefig(save_path_test)
plt.clf()

# Plotting results for train success rate
plt.figure(figsize=(10, 6))

for i in range(3):
    plt.plot(range(len(models_average_values_train[i])),
             models_average_values_train[i],
             label=labels[i], color=colors[i])
    plt.fill_between(range(len(models_average_values_train[i])),
                     models_average_values_train[i] - models_standard_devs_train[i],
                     models_average_values_train[i] + models_standard_devs_train[i],
                     color=colors[i], alpha=0.2)

plt.title(train_plot_title)
plt.xlabel('Episode')
plt.ylabel('Train Success Rate')
plt.legend()
save_path_train = os.path.join("log", f"{instance_id}_train")
plt.savefig(save_path_train)
