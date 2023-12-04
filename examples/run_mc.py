import subprocess
# List of student-teacher configurations
configs = [
    "coffee-button-v2-goal-observable,coffee-button-v2-goal-observable",
    "plate-slide-v2-goal-observable,plate-slide-v2-goal-observable"
]

# List of seeds to run through
seeds = [0, 2, 3, 4]


# Counter for alternating between GPUs
counter = 0

# Iterate through each configuration
for config in configs:
    # Iterate through each seed
    for seed in seeds:
        # Extract the guide, secondary guide, and student environments from the configuration string
        envs = config.split(',')
        guide_env = envs[0]
        student_envs = ','.join(envs[1:])
        # Create a tmux session for the current configuration
        # Truncate the "-v2-goal-observable" suffix from the environment names
        session_name = f"{seed}_{guide_env.replace('-v2-goal-observable', '')}_{student_envs.replace('-v2-goal-observable', '')}_monte_carlo"
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name])
        # Run the jsrl conda env and cd into examples folder
        cmd = f"conda activate jsrl && cd /home/ubuntu/jjlee/jumpstart-rl/examples"
        subprocess.run(["tmux", "send-keys", "-t", session_name, cmd, "Enter"])
        # Run the training script with the current configuration flags
        # Add CUDA device flag to command
        guides_directory = f"/home/ubuntu/jjlee/jumpstart-rl/models/{guide_env}_guide_sac/best_model"
        cmd = f"python train_jsrl_mc_side_by_side.py --sparse=True --guide_env={guide_env} --student_env={student_envs} --seed={seed} --timesteps=1000000 --wandb=True --guides_directory={guides_directory} --log_true_q=True"
        subprocess.run(["tmux", "send-keys", "-t", session_name, cmd, "Enter"])
        # Increment counter
        counter += 1