import argparse
import gymnasium as gym
from gym.core import Env
from gymnasium.core import Wrapper
from gymnasium.wrappers import TimeLimit
from datetime import date

import sys
sys.path.append('/home/ubuntu/jjlee/jumpstart-rl/')

import os
os.environ["TQDM_DISABLE"] = "1"

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np
from src.jsrl.jsrl import get_jsrl_algorithm, JSRLGuides
import wandb
from wandb.integration.sb3 import WandbCallback

from typing import Union

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

class MetaWorldWrapper(Wrapper):
    def __init__(self, env: Env, sparse=True, truncate_when_done=False):
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
            truncated = bool(info["success"]) or truncated
            
        return obs, reward, terminated, truncated, info
    
    
    def reset(self, seed=None):
        return self.env.reset(seed=seed)

def main(seed, env_name, guide_steps, timesteps, student_env, strategy, grad_steps, sparse, data_collection_strategy, epsilon, use_wandb, learning_starts=30000, guides_directory=None, priority=False, log_true_q=False, cuda="cuda:0"):
    if student_env is None:
        student_env = env_name

    use_wandb = use_wandb == "True"
    priority = priority == "True"
    log_true_q = log_true_q == "True"
    

    cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[student_env]
    env = cls(seed=seed)
    env._freeze_rand_vec = False   
    
    # env = TimeLimit(env, env.max_path_length)
    env = MetaWorldWrapper(env, sparse=sparse, truncate_when_done=False)
    env = DummyVecEnv([lambda: env])
    
    eval_env = cls(seed=1)
    eval_env._freeze_rand_vec = False
    # eval_env = TimeLimit(eval_env, eval_env.max_path_length)
    eval_env = MetaWorldWrapper(eval_env, sparse=sparse, truncate_when_done=True)
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # guides_directory is currently a string, but it should be a list of strings. Decode the guides_directory string into a list of strings
    if guides_directory is not None:
        guides_directory = guides_directory.split(",")
        
    
    guide_policy = JSRLGuides(
        guides_directory=guides_directory,
        priority=priority
    )
    
    guide_env=[]
    
    # Extract the environments that guides were trained in from the guides directory
    # The guide directory will be in the following format: /ext_hdd/jjlee/jumpstart-rl/examples/models/{env_name}_guide_{model_name}_{'sparse' if sparse else 'dense'}
    if guides_directory is not None:
        for guide_directory in guides_directory:
            env_name = guide_directory.split("/")[-2].split("_guide_")[0]
            guide_env.append(env_name)

    config = {
        "policy_type": strategy,
        "problem": "multi_task",
        "guide_steps": guide_steps,
        "total_timesteps": 25000,
        "guide_env_name": guide_env,
        "student_env_name": f"{student_env}",
        "seed": seed,
        "grad_steps": grad_steps,
        "sparse": sparse,
        "data_collection_strategy": data_collection_strategy,
        "epsilon": epsilon,
        "learning_starts": learning_starts,
        "priority": priority,
        "log_true_q": log_true_q,
        "student_model": "TD3",
    }
    if use_wandb==True:
        wandb.init(project="jsrl",
                entity="jsrl-boys",
            config=config,
            group=f"{date.today()}",
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True, 
        )
    
    n = 10
    #set max_horizon dynamically according to the environment
    if env_name == "PointMaze_UMaze-v3":
        max_horizon = 60
    else:
        max_horizon = 300
        
    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.7 * np.ones(n_actions))
    
    comparison_model = TD3(
        "MlpPolicy",  # You can also use "CnnPolicy" for CNN architectures
            env=env,
            verbose=1,
            tensorboard_log=f"/home/ubuntu/jjlee/jumpstart-rl/logs/{env_name}_guide_TD3",
            seed=seed,
            learning_starts=1000,
            action_noise=action_noise,
    )
    
    model = get_jsrl_algorithm(TD3)(
        "MlpPolicy",
        env=env,
        policy_kwargs=dict(
            guide_policy=guide_policy,
            strategy=strategy,
            horizons=np.arange(max_horizon, -1, -max_horizon // n,),
            eval_freq=10000,
            n_eval_episodes=10,
        ),
        data_collection_strategy=data_collection_strategy,
        verbose=0,
        seed=seed,
        eval_env=eval_env,
        gradient_steps=grad_steps,
        device=cuda,
        learning_starts=learning_starts,
        epsilon=epsilon,
        log_true_q=log_true_q,
        comparison_model=comparison_model,
        # action_noise=action_noise
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
    parser.add_argument("--data_collection_strategy", type=str, default="multi", help="The data collection strategy to use")
    parser.add_argument("--epsilon", type=float, default=0, help="The epsilon value to use")
    parser.add_argument("--learning_starts", type=int, default=0, help="The number of steps to delay the teacher help")
    parser.add_argument("--wandb", type=str, default="False", help="Whether to use wandb")
    parser.add_argument("--guides_directory", type=str, default="/home/ubuntu/jjlee/jumpstart-rl/models/coffee-button-v2-goal-observable_guide_sac/best_model.zip", help="The directory of the guides")
    parser.add_argument("--priority", type=str, default="False", help="Whether to use priority")
    parser.add_argument("--log_true_q", type=str, default="True", help="Whether to log the true q values")
    parser.add_argument("--cuda_device", type=str, default="cuda:0", help="The cuda device to use")
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
        learning_starts=args.learning_starts,
        guides_directory=args.guides_directory,
        priority=args.priority,
        log_true_q=args.log_true_q,
        cuda=args.cuda_device
    )
        
