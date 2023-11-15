#!/bin/bash

# Get into the directory where this script is located
cd "/home/jjlee/jumpstart-rl/examples"

# List of student-teacher configurations
configs=(
    "coffee-button-v2-goal-observable"
    "plate-slide-v2-goal-observable"
)

# Iterate through each configuration
for config in "${configs[@]}"
do
    # Extract the guide and student environments from the configuration string
    IFS=',' read -ra envs <<< "$config"
    guide_env="${envs[0]}"


    # Create a tmux session for the current configuration
    # Truncate the "-v2-goal-observable" suffix from the environment names
    session_name="3_${guide_env//"-v2-goal-observable"/}_MC"
    tmux new-session -d -s "$session_name"

    # Run the training script with the current configuration flags
    cmd="python train_guide.py --sparse=True --env=${guide_env} --model=td3 --timesteps=1000000 --grad_steps=1 --seed=3 --use_wandb=True --log_true_q=True"
    tmux send-keys -t "$session_name" "$cmd" Enter
done