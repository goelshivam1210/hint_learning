#!/bin/bash

# Create a new tmux session
tmux new-session -d -s simple

# Start the first process in the first pane
tmux send-keys -t simple "python run.py --use_smallenv --seed 0 --device cuda" C-m

# Create panes and run the processes with a small delay
for seed in {1..4}; do
    sleep 0.5  # Add a short delay (500ms) to prevent timestamp conflicts
    tmux split-window -t simple -h "python run.py --use_smallenv --seed $seed --device cuda"
    tmux select-layout -t simple tiled  # Arrange panes in a tiled layout
done

# Attach to the session
tmux attach-session -t simple