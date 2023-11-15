#!/bin/bash

# Get into the directory where this script is located
cd "/home/jjlee/jumpstart-rl/examples"

# List of student-teacher configurations
configs=(
    "plate-slide-v2-goal-observable,plate-slide-side-v2-goal-observable"
)

guides_directory="/ext_hdd/jjlee/jumpstart-rl/examples/models/plate-slide-v2-goal-observable_guide_sac/best_model.zip,/ext_hdd/jjlee/jumpstart-rl/examples/models/coffee-button-v2-goal-observable_guide_sac/best_model.zip"

# Iterate through each configuration
for config in "${configs[@]}"
do
    # Extract the guide, secondary guide, and student environments from the configuration string
    IFS=',' read -ra envs <<< "$config"
    guide_env="${envs[0]}"
    student_envs="${envs[@]:1}"
    # Create a tmux session for the current configuration
    # Truncate the "-v2-goal-observable" suffix from the environment names
    session_name="${guide_env//"-v2-goal-observable"/}_${student_envs//"-v2-goal-observable"/}_multi"
    tmux new-session -d -s "$session_name"

    # Run the training script with the current configuration flags
    cmd="python train_jsrl_curriculum_multi_teacher.py --sparse=True --guide_env=$guide_env --student_env=$student_envs --seed=0 --timesteps=500000 --data_collection_strategy=multi --wandb=True --guides_directory=$guides_directory --priority=False"
    tmux send-keys -t "$session_name" "$cmd" Enter
done