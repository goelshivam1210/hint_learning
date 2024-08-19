# See-Say : Code base for Hint guided learning

Currently we have custom gridworld environment using the MiniGrid framework that trains a reinforcement learning agent using Proximal Policy Optimization (PPO) with Tianshou. The goal of the agent is to collect resources, craft a sword, and find a treasure in the environment.

## Table of Contents
1. [Overview](#overview)
2. [Environment Details](#environment-details)
3. [Actions](#actions)
4. [Observation Space](#observation-space)
5. [Reward Structure](#reward-structure)
6. [Training Details](#training-details)
7. [Manual Control](#manual-control)
8. [Installation](#installation)
9. [Usage](#usage)

## Overview

The agent operates in a gridworld containing resources (Iron Ore, Silver Ore, Platinum Ore, Gold Ore, Trees), a crafting table, and a chest. The objective of the agent is to collect resources, craft a sword, and then open the chest to find the treasure. The environment is built using `MiniGridEnv` from MiniGrid, and the agent is trained using the PPO algorithm from Tianshou.

## Environment Details

The environment is a 12x12 grid, with the agent starting at a random or specified position. It contains the following objects:

- **Iron Ore** (red)
- **Silver Ore** (grey)
- **Platinum Ore** (purple)
- **Gold Ore** (yellow)
- **Tree** (green)
- **Chest** (purple)
- **Crafting Table** (blue)
- **Walls**

The agent uses LiDAR to detect nearby objects in the environment, and it has an inventory to store collected resources.

## Actions

The agent can perform the following actions:

1. `move_forward`: Move the agent forward by one step.
2. `turn_left`: Rotate the agent 90 degrees to the left.
3. `turn_right`: Rotate the agent 90 degrees to the right.
4. `toggle`: Interact with objects (collect resources or interact with boxes).
5. `craft_sword`: Craft a sword using resources in the inventory.
6. `open_chest`: Open the chest to win the game if the agent has crafted a sword.

## Observation Space

The observation space consists of:

- **LiDAR**: A grid with 8 beams, each detecting one of the 8 possible objects in the environment. 
- **Inventory**: The agent’s inventory containing resources it has collected.

The LiDAR data is flattened and concatenated with the inventory data to form the final observation space.

## Reward Structure

- **Per-Step Penalty**: -1 for each time step to encourage faster completion of the task.
- **Resource Collection**: +1 for collecting a new resource.
- **Crafting the Sword**: +50 (only during the first `max_reward_episodes` episodes).
- **Opening the Chest**: +1000 for successfully opening the chest after crafting the sword.
- **Failures**: -1 for attempting invalid actions (e.g., crafting without the necessary resources).

## Training Details

The training is implemented using the PPO algorithm from Tianshou. The agent is trained across 8 parallel environments, using a replay buffer to store experience. The policy and value networks are trained jointly using the collected experience.

### Success Criteria

The agent's training will stop once it reaches a success rate of 90% over the last 10 evaluations. Success is defined as successfully opening the chest.

## Manual Control

The environment also supports manual control via keyboard inputs. You can control the agent using the following keys:

- **Arrow Keys**: Move the agent (left, right, up).
- **Spacebar**: Interact with objects (collect resources).
- **C**: Craft the sword.
- **O**: Open the chest.

## Installation

### Requirements

- Python 3.8+
- Gymnasium
- MiniGrid
- Tianshou
- Pytorch
- Numpy
- TensorBoard

### Installation Steps

1. Clone the repository.



2. Create a virtual environment and activate it:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
    or Conda
    
    ```bash
    conda create --name see_say python=3.11.4
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Agent

To train the agent using PPO, run the following command:

```bash
python train_on_policy.py
```

This will start the training process and log results to TensorBoard.

### Running Manual Control

To manually control the agent, run the following command:

```bash
python env.py
```
You can then control the agent with the keyboard as described in the Manual Control section.

</br>

## More to come soon! Stay tuned!!! 
