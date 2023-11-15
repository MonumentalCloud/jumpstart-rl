import subprocess

# List of student-teacher configurations
configs = [
    "coffee-button-v2-goal-observable",
    "plate-slide-v2-goal-observable"
]

# List of seeds to run through
seeds = [0, 2, 3, 4]

# Iterate through each configuration
for config in configs:
    # Iterate through each seed
    for seed in seeds:
        # Extract the guide environment from the configuration string
        guide_env = config

        # Create a tmux session for the current configuration
        # Truncate the "-v2-goal-observable" suffix from the environment names
        session_name = f"{seed}_{guide_env.replace('-v2-goal-observable', '')}_MC"
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name])

        # Run the training script with the current configuration flags
        cmd = f"python train_guide.py --sparse=True --env={guide_env} --model=td3 --timesteps=1000000 --grad_steps=1 --seed={seed} --use_wandb=True --log_true_q=True"
        subprocess.run(["tmux", "send-keys", "-t", session_name, cmd, "Enter"])