# This is the same as train_jsrl_curriculum.py, except that the strategy is "threshold" instead of "curriculum".
# The "threshold" strategy is the same as the "curriculum" strategy, except that the JSRL policy is not dependent on the guide policy at every timestep, but only when the confidence of the exploration policy is below a certain threshold.
# The threshold is set to 0.5 by default, but can be changed by passing the threshold argument to the JSRL policy.

import argparse
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
from src.jsrl.jsrl_confidence_threshold import get_jsrl_algorithm
from src.jsrl.RND import RNDEstimator
import wandb
from wandb.integration.sb3 import WandbCallback

from typing import Union



def main(seed, env_name, threshold, guide_steps):
    env = gym.make(env_name, continuing_task=False, max_episode_steps=150)
    
    if guide_steps == "best_model":
        guide_policy = TD3.load(f"/ext_hdd/jjlee/jumpstart-rl/examples/models/{env_name}_TD3/best_model").policy
    else:
        guide_policy = TD3.load(f"/ext_hdd/jjlee/jumpstart-rl/examples/models/{env_name}_TD3/{guide_steps}steps_model.zip").policy
    
    config = {
        "policy_type": "threshold",
        "guide_steps": guide_steps,
        "total_timesteps": 25000,
        "env_name": f"{env_name}",
        "threshold": threshold
    }
    
    wandb.init(project="jsrl",
        group=f"threshold_{guide_steps}_steps",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True, 
    )
    n = 10
    max_horizon = 60
    uncertainty_estimator = RNDEstimator(env.observation_space['observation'].shape[0], env.action_space.shape[0])
    model = get_jsrl_algorithm(TD3, uncertainty_estimator=uncertainty_estimator, threshold=threshold)(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict(
            guide_policy=guide_policy,
            max_horizon=max_horizon,
            strategy="threshold",
            horizons=np.arange(max_horizon, -1, -max_horizon // n,)
        ),
        verbose=1,
        tensorboard_log="ext_hdd/jjlee/jumpstart-rl/logs/pointmaze_jsrl_threshold",
        seed=seed
    )
    
    model.learn(
        total_timesteps=1e5,
        log_interval=10,
        progress_bar=True,
        callback=WandbCallback(
            gradient_save_freq=100,
            # model_save_path="examples/models/pointmaze_jsrl_threshold_TD3",
            verbose=2,
        )
        # callback=EvalCallback(
        #     env,
        #     n_eval_episodes=100,
        #     best_model_save_path="examples/models/pointmaze_jsrl_threshold_TD3"
        # ),
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--env", type=str, default="PointMaze_UMaze-v3", help="Environment name")
    parser.add_argument("--threshold", type=float, default=0.01, help="Threshold for confidence")
    parser.add_argument("--guide_steps", type=str, default="best_model", help="Number of steps (Expertise) of the guide policy")
    args = parser.parse_args()
    main(args.seed, args.env, args.threshold, args.guide_steps)