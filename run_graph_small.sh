#!/bin/bash

# Create a new tmux session
tmux new-session -d -s graph_small

# Start the first process in the first pane
tmux send-keys -t graph_small "python run.py --use_wrapper --use_smallenv --use_graph_reward --seed 0" C-m

# Create panes and run the processes with a small delay
for seed in {1..4}; do
    sleep 1  # Add a short delay (500ms) to prevent timestamp conflicts
    tmux split-window -t graph_small -h "python run.py --use_wrapper --use_smallenv --use_graph_reward --seed $seed"
    tmux select-layout -t graph_small tiled  # Arrange panes in a tiled layout
done

# Attach to the session
tmux attach-session -t graph_small