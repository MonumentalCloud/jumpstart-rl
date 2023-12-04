# import gymnasium as gym
from gym.core import Env
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import random
from gymnasium.wrappers import TimeLimit
from gymnasium.core import Wrapper
import gym
from stable_baselines3 import TD3, PPO, SAC, DQN, DDPG
#import makevecenv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.vec_env import VecExtractDictObs
from sb3_contrib import TQC
import argparse
import os
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np

import sys
sys.path.append('/home/ubuntu/jjlee/jumpstart-rl/')

from src.jsrl.guide_helper import get_algorithm

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# class CustomEvalCallback(EvalCallback)
#     def __init__(self, env, n_eval_episodes=5, eval_freq=10000, best_model_save_path=None, log_path=None, model_save_path=None):
#         super().__init__(eval_env=env, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq, best_model_save_path=best_model_save_path, log_path=log_path)
#         self.model_save_path = model_save_path

#     def _on_step(self) -> bool:
#         # if timestep is a multiple of 20000, save the model
#         if self.n_calls % 20000 == 0:
#             self.model.save(os.path.join(self.model_save_path, f"{self.n_calls}steps_model.zip"))

# Write a custom env wrapper that wraps around a metaworld env so that it converts info['success'] into info['is_success'] so that it works with the stable-baselines3 eval callback
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

def main(env_name, timesteps, model_name, grad_steps, sparse, log_true_q, use_wandb, seed, cuda_device):
    
    sparse = sparse == "True"
    use_wandb = use_wandb == "True"
    
    cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
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
    
    # env.observation_space.spaces = env.observation_space
    
    
    model_classes = {
        "td3": TD3,
        "ppo": PPO,
        "sac": SAC,
        "dqn": DQN,
        "ddpg": DDPG,
        "tqc": TQC,
        
    }
    model_class = model_classes[model_name.lower()]
    
    config = {
        "policy_type": "guide",
        "total_timesteps": 25000,
        "guide_env_name": f"{env_name}",
        "gradient_steps": grad_steps,
        "sparse": sparse,
        "log_true_q": log_true_q,
    }
    
    if use_wandb:
        wandb.init(project="jsrl",
                entity="jsrl-boys",
            dir="/home/ubuntu/jjlee/jumpstart-rl/logs",
            config=config,
            group=f"guide_{model_name}_{env_name}",
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True, 
        )
    
    
    if model_name == "tqc":
        # Follow the following specification from the yml file in order to use HER with tqc:
        # FetchPush-v1:
        # env_wrapper:
        #     - sb3_contrib.common.wrappers.TimeFeatureWrapper
        # n_timesteps: !!float 1e6
        # policy: 'MlpPolicy'
        # model_class: 'tqc'
        # n_sampled_goal: 4
        # goal_selection_strategy: 'future'
        # buffer_size: 1000000
        # batch_size: 2048
        # gamma: 0.95
        # learning_rate: !!float 1e-3
        # tau: 0.05
        # policy_kwargs: "dict(n_critics=2, net_arch=[512, 512, 512])"
        # online_sampling: True
        
        # Define the HER parameters
        her_kwargs = {
            "goal_selection_strategy": "future",
        }
        
        model = get_algorithm(TQC)(
            "MlpPolicy",  # You can also use "CnnPolicy" for CNN architectures
            env=env,
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.98,
            tensorboard_log=f"/home/ubuntu/jjlee/jumpstart-rl/logs/{env_name}_guide_{model_name}",
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=her_kwargs,
            learning_starts=1000,
            gradient_steps=grad_steps,
            log_true_q=log_true_q, 
            eval_env=eval_env,
            eval_freq=10000,
            n_eval_episodes=10,
            seed=seed,
            device=cuda_device
        )
        
    elif model_name == "ddpg":
        # Define the HER parameters
        her_kwargs = {
            "goal_selection_strategy": "future",
        }
        
        model = get_algorithm(DDPG)(
            "MlpPolicy",  # You can also use "CnnPolicy" for CNN architectures
            env=env,
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.98,
            tensorboard_log=f"/home/ubuntu/jjlee/jumpstart-rl/logs/{env_name}_guide_{model_name}",
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=her_kwargs,
            learning_starts=1000,
            gradient_steps=grad_steps,
            log_true_q=log_true_q,
            eval_env=eval_env,
            eval_freq=10000,
            n_eval_episodes=10,
            seed=seed,
            device=cuda_device
        )
    
    elif model_name == "sac":
        # Tune the hyperparameters of SAC so that it will perform optimally in the metaworld environments, especially for dial-turn-v2-goal-observable
        
        # Description	value	variable_name
        # Normal Hyperparameters
        # Batch size	500	batch_size
        # Number of epochs	500	n_epochs
        # Path length per roll-out	500	max_path_length
        # Discount factor	0.99	discount
        # Algorithm-Specific Hyperparameters
        # Policy hidden sizes	(256,256)	hidden_sizes
        # Activation function of hidden layers	ReLU	hidden_nonlinearity
        # Policy learning rate	3×10−4	policy_lr
        # Q-function learning rate	3×10−4	qf_lr
        # Policy minimum standard deviation	e−20	min_std
        # Policy maximum standard deviation	e2	max_std
        # Gradient steps per epoch	500	gradient_steps_per_itr
        # Number of epoch cycles	40	epoch_cycles
        # Soft target interpolation parameter	5×10−3	target_update_tau
        # Use automatic entropy Tuning	True	use_automatic_entropy_tuning

        model = get_algorithm(SAC)(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=f"/home/ubuntu/jjlee/jumpstart-rl/logs/{env_name}_guide_{model_name}",
            learning_rate=3e-4,
            buffer_size=int(1e6),
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=grad_steps,
            action_noise=None,
            optimize_memory_usage=False,
            target_update_interval=1,
            target_entropy="auto",
            use_sde=False,
            sde_sample_freq=-1,
            policy_kwargs=None,
            eval_env=eval_env,
            log_true_q=log_true_q,
            eval_freq=10000,
            n_eval_episodes=10,
            seed=seed,
            device=cuda_device
        )
    else:
        # The noise objects for TD3
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))    
        
        model = get_algorithm(model_class)(
            "MlpPolicy",  # You can also use "CnnPolicy" for CNN architectures
            env=env,
            verbose=1,
            tensorboard_log=f"/home/ubuntu/jjlee/jumpstart-rl/logs/{env_name}_guide_{model_name}",
            log_true_q=log_true_q,
            eval_env=eval_env,
            seed=seed,
            eval_freq=10000,
            n_eval_episodes=10,
            learning_starts=1000,
            action_noise=action_noise,
            device=cuda_device
        )
        
    if use_wandb:
        callback = [WandbCallback(
            gradient_save_freq=10000,
            # model_save_path=f"/ext_hdd/jjlee/jumpstart-rl/examples/models/{env_name}_guide_{model_name}_{'sparse' if sparse else 'dense'}",
            verbose=2,
        ),
                  EvalCallback(
            eval_env,
            n_eval_episodes=100,
            best_model_save_path=f"/home/ubuntu/jjlee/jumpstart-rl/models/{env_name}_guide_{model_name}_{'sparse' if sparse else 'dense'}"
        ),]
    else:
        callback = [EvalCallback(
            eval_env,
            n_eval_episodes=100,
            best_model_save_path=f"/home/ubuntu/jjlee/jumpstart-rl/models/{env_name}_guide_{model_name}_{'sparse' if sparse else 'dense'}"
        ),
        ]
    
    model.learn(
        total_timesteps=timesteps,
        log_interval=10,
        progress_bar=True,
        # callback=CustomEvalCallback(
        #     env,
        #     n_eval_episodes=100,
        #     eval_freq=10000,
        #     model_save_path=f"examples/models/{env_name}_TD3"
        # )
        callback=callback,
        # callback=EvalCallback(
        #     env,
        #     n_eval_episodes=100,
        #     best_model_save_path=f"/ext_hdd/jjlee/jumpstart-rl/examples/models/{env_name}_guide_{model_name}"
        # ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="coffee-button-v2-goal-observable", help="Environment name")
    parser.add_argument("--timesteps", type=int, default=1e6)
    parser.add_argument("--model", type=str, default="sac")
    parser.add_argument("--grad_steps", type=int, default=1)
    parser.add_argument("--sparse", type=bool, default=False)
    parser.add_argument("--log_true_q", type=bool, default="True")
    parser.add_argument("--use_wandb", type=str, default="False")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda_device", type=str, default="cuda:0")
    args = parser.parse_args()
    main(args.env, args.timesteps, args.model, args.grad_steps, args.sparse, args.log_true_q, args.use_wandb, args.seed, args.cuda_device)