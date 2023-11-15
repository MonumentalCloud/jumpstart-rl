#!/bin/bash

# Get into the directory where this script is located
cd "/home/jjlee/jumpstart-rl/examples"

# List of student-teacher configurations
configs=(
    "coffee-button-v2-goal-observable,coffee-button-v2-goal-observable"
    "plate-slide-v2-goal-observable,plate-slide-v2-goal-observable"
)

# Iterate through each configuration
for config in "${configs[@]}"
do
    # Extract the guide and student environments from the configuration string
    IFS=',' read -ra envs <<< "$config"
    guide_env="${envs[0]}"
    student_envs="${envs[@]:1}"

    # Create a tmux session for the current configuration
    # Truncate the "-v2-goal-observable" suffix from the environment names
    session_name="2_${guide_env//"-v2-goal-observable"/}_${student_envs//"-v2-goal-observable"/}_success"
    tmux new-session -d -s "$session_name"

    # Run the training script with the current configuration flags
    cmd="python train_jsrl_curriculum_multi_task.py --sparse=True --guide_env=$guide_env --student_env=$student_envs --seed=5 --timesteps=500000 --wandb=True --epsilon=0.2"
    tmux send-keys -t "$session_name" "$cmd" Enter
done