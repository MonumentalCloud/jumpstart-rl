import argparse
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback

import numpy as np
from src.jsrl.jsrl import get_jsrl_algorithm
import wandb
from wandb.integration.sb3 import WandbCallback

from typing import Union

def main(seed, env_name, guide_steps, timesteps, student_env):
    if student_env is None:
        student_env = env_name
    env = gym.make(student_env, max_episode_steps=150) #continuing_task=False)
    if guide_steps == "best_model":
        guide_policy = TD3.load(f"/ext2/jjlee/jumpstart-rl/models/{env_name}_guide_TD3/best_model").policy
    else:
        guide_policy = TD3.load(f"/ext2/jjlee/jumpstart-rl/models/{env_name}_TD3/{guide_steps}steps_model.zip").policy
    
    config = {
        "policy_type": "curriculum",
        "guide_steps": guide_steps,
        "total_timesteps": 25000,
        "guide_env_name": f"{env_name}",
        "student_env_name": f"{student_env}",
        "seed": seed,
    }
    
    wandb.init(project="jsrl",
        config=config,
        group=f"curriculum_{guide_steps}_steps",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True, 
    )
    
    n = 10
    #set max_horizon dynamically according to the environment
    if env_name == "PointMaze_UMaze-v3":
        max_horizon = 60
    else:
        max_horizon = 150
    model = get_jsrl_algorithm(TD3)(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict(
            guide_policy=guide_policy,
            max_horizon=max_horizon,
            strategy="curriculum",
            horizons=np.arange(max_horizon, -1, -max_horizon // n,)
        ),
        verbose=1,
        tensorboard_log="ext_hdd/jjlee/jumpstart-rl/logs/pointmaze_jsrl_curriculum",
        seed=seed
    )
    model.learn(
        total_timesteps=timesteps,
        log_interval=10,
        progress_bar=True,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path="ext_hdd/jjlee/jumpstart-rl/examples/models/pointmaze_jsrl_curriculum_TD3",
            verbose=2,
        )
        # callback=EvalCallback(
        #     env,
        #     n_eval_episodes=100,
        #     best_model_save_path="examples/models/pointmaze_jsrl_curriculum_TD3"
        # ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--env", type=str, default="PointMaze_UMaze-v3", help="Environment name")
    parser.add_argument("--guide_steps", type=str, default="best_model", help="Number of steps (Expertise) of the guide policy")
    parser.add_argument("--timesteps", type=int, default=1e5, help="Number of timesteps to train the agent for")
    parser.add_argument("--student_env", type=str, default=None, help="Environment name")
    args = parser.parse_args()
    main(args.seed, args.env, args.guide_steps, args.timesteps, args.student_env)
