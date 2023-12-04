import subprocess

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
# Get all the environments from metaworld
envs = list(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys())

# List of seeds to run through
seeds = [0, 2, 3, 4]

# Counter for alternating between GPUs
counter = 0

# Iterate through each environment
for env in envs:
    # Create a tmux session for the current configuration
    session_name = f"{env}_guide"
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name])
    # Run the jsrl conda env and cd into examples folder
    cmd = f"conda activate jsrl && cd /home/ubuntu/jjlee/jumpstart-rl/examples"
    subprocess.run(["tmux", "send-keys", "-t", session_name, cmd, "Enter"])
    # Run the training script with the current configuration flags
    # Add CUDA device flag to command
    cmd = f"python train_guide.py --sparse=False --timesteps=500000 --use_wandb=True --model=sac --env={env} --log_true_q=True"
    subprocess.run(["tmux", "send-keys", "-t", session_name, cmd, "Enter"])
    # Increment counter
    counter += 1