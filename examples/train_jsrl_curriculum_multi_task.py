import argparse
import gymnasium as gym
from gym.core import Env
from gymnasium.core import Wrapper
from gymnasium.wrappers import TimeLimit
from datetime import date

import sys
sys.path.append('/home/jjlee/jumpstart-rl/')

import os
os.environ["TQDM_DISABLE"] = "1"

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np
from src.jsrl.jsrl import get_jsrl_algorithm
import wandb
from wandb.integration.sb3 import WandbCallback

from typing import Union


class MetaWorldWrapper(Wrapper):
    def __init__(self, env: Env, sparse=False, truncate_when_done=False):
        super().__init__(env)
        self.env = env
        self.sparse = sparse
        self.truncate_when_done = truncate_when_done
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['is_success'] = info['success']
        if self.sparse:
            reward=int(info['success'])
        if self.truncate_when_done:
            truncated = info["success"] or truncated
            
        return obs, reward, terminated, truncated, info
    
    
    def reset(self, seed=None):
        return self.env.reset(seed=seed)

def main(seed, env_name, guide_steps, timesteps, student_env, strategy, grad_steps, sparse, data_collection_strategy, epsilon, use_wandb, delay_steps):
    if student_env is None:
        student_env = env_name

    use_wandb = use_wandb == "True"
 
    cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[student_env]
    env = cls(seed=seed)
    env._freeze_rand_vec = False   
    
    # env = TimeLimit(env, env.max_path_length)
    env = MetaWorldWrapper(env, sparse=sparse, truncate_when_done=sparse)
    env = DummyVecEnv([lambda: env])
    
    eval_env = cls(seed=1)
    eval_env._freeze_rand_vec = False
    # eval_env = TimeLimit(eval_env, eval_env.max_path_length)
    eval_env = MetaWorldWrapper(eval_env, sparse=sparse, truncate_when_done=True)
    eval_env = DummyVecEnv([lambda: eval_env])
    
    if guide_steps == "best_model":
        guide_policy = SAC.load(f"/ext_hdd/jjlee/jumpstart-rl/examples/models/{env_name}_guide_sac/best_model", device="cuda:1").policy
    else:
        guide_policy = SAC.load(f"/ext_hdd/jjlee/jumpstart-rl/examples/models/{env_name}_sac/{guide_steps}steps_model.zip", device="cuda:1").policy
    
    config = {
        "policy_type": strategy,
        "problem": "multi_task",
        "guide_steps": guide_steps,
        "total_timesteps": 25000,
        "guide_env_name": f"{env_name}",
        "student_env_name": f"{student_env}",
        "seed": seed,
        "grad_steps": grad_steps,
        "sparse": sparse,
        "data_collection_strategy": data_collection_strategy,
        "epsilon": epsilon,
        "delay_steps": delay_steps,
    }
    if use_wandb==True:
        print(use_wandb)
        wandb.init(project="jsrl",
                entity="jsrl-boys",
            config=config,
            group=f"{date.today()}",
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,
            dir="/ext_hdd/jjlee/jumpstart-rl/logs/pointmaze_jsrl_curriculum", 
        )
    
    n = 10
    #set max_horizon dynamically according to the environment
    if env_name == "PointMaze_UMaze-v3":
        max_horizon = 60
    else:
        max_horizon = 300
    model = get_jsrl_algorithm(SAC)(
        "MlpPolicy",
        env=env,
        policy_kwargs=dict(
            guide_policy=guide_policy,
            strategy=strategy,
            horizons=np.arange(max_horizon, -1, -max_horizon // n,),
            eval_freq=10000,
            n_eval_episodes=10,
            data_collection_strategy=data_collection_strategy,
            epsilon=epsilon,
            delay_steps=delay_steps,
        ),
        verbose=0,
        tensorboard_log="ext_hdd/jjlee/jumpstart-rl/logs/pointmaze_jsrl_curriculum",
        seed=seed,
        eval_env=eval_env,
        gradient_steps=grad_steps,
        device="cuda:1",
    )
    callback = []
    if use_wandb == True:
        callback = [WandbCallback(
        ),]
    model.learn(
        total_timesteps=timesteps,
        log_interval=10,
        progress_bar=True,
        callback=callback,
        #           EvalCallback(
        #     eval_freq=100,
        #     eval_env=eval_env,
        #     n_eval_episodes=5,
        #     # best_model_save_path=f"/ext_hdd/jjlee/jumpstart-rl/examples/models/{env_name}_to_{student_env}_curriculum_sac"
        # ),]
        # callback=EvalCallback(
        #     env,
        #     n_eval_episodes=100,
        #     best_model_save_path="examples/models/pointmaze_jsrl_curriculum_TD3"
        # ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--guide_env", type=str, default="coffee-button-v2-goal-observable", help="Environment name")
    parser.add_argument("--guide_steps", type=str, default="best_model", help="Number of steps (Expertise) of the guide policy")
    parser.add_argument("--timesteps", type=int, default=1e6, help="Number of timesteps to train the agent for")
    parser.add_argument("--student_env", type=str, default="coffee-button-v2-goal-observable", help="Environment name")
    parser.add_argument("--strategy", type=str, default="curriculum", help="The strategy to use")
    parser.add_argument("--grad_steps", type=int, default=1, help="Number of gradient steps to take")
    parser.add_argument("--sparse", type=bool, default=True, help="Whether to use sparse rewards")
    parser.add_argument("--data_collection_strategy", type=str, default="normal", help="The data collection strategy to use")
    parser.add_argument("--epsilon", type=float, default=0.1, help="The epsilon value to use")
    parser.add_argument("--delay_steps", type=int, default=0, help="The number of steps to delay the teacher help")
    parser.add_argument("--wandb", type=str, default="False", help="Whether to use wandb")
    args = parser.parse_args()
    main(
        seed=args.seed,
        env_name=args.guide_env,
        guide_steps=args.guide_steps,
        timesteps=args.timesteps,
        student_env=args.student_env,
        strategy=args.strategy,
        grad_steps=args.grad_steps,
        sparse=args.sparse,
        data_collection_strategy=args.data_collection_strategy,
        epsilon=args.epsilon,
        use_wandb=args.wandb,
        delay_steps=args.delay_steps,
    )
        