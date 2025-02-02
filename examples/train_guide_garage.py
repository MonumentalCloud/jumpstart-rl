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
    def __init__(self, env: Env):
        super().__init__(env)
        self.env = env
        
    def step(self, action):
        obs, reward, _, done, info = self.env.step(action)
        info['is_success'] = info['success']
        return obs, reward, _, done, info
    
    def reset(self, seed=None):
        return self.env.reset(seed=seed)


def main(env_name, timesteps, model_name):
    
    cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
    env = cls()
    env._freeze_rand_vec = False   
    
    env = TimeLimit(env, env.max_path_length)
    env = MetaWorldWrapper(env)
    env = DummyVecEnv([lambda: env])
    
    eval_env = cls(seed=1)
    eval_env._freeze_rand_vec = False
    eval_env = TimeLimit(eval_env, eval_env.max_path_length)
    eval_env = MetaWorldWrapper(eval_env)
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
    }
    
    wandb.init(project="jsrl",
        dir="/ext_hdd/jjlee/jumpstart-rl/logs",
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
        
        model = TQC(
            "MlpPolicy",  # You can also use "CnnPolicy" for CNN architectures
            env=env,
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.98,
            tensorboard_log=f"/ext_hdd/jjlee/jumpstart-rl/logs/{env_name}_guide_{model_name}",
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=her_kwargs,
            learning_starts=1000,
        )
    elif model_name == "ddpg":
        # Define the HER parameters
        her_kwargs = {
            "goal_selection_strategy": "future",
        }
        
        model = DDPG(
            "MlpPolicy",  # You can also use "CnnPolicy" for CNN architectures
            env=env,
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.98,
            tensorboard_log=f"/ext_hdd/jjlee/jumpstart-rl/logs/{env_name}_guide_{model_name}",
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=her_kwargs,
            learning_starts=1000,
        )

    else:    
        model = model_class(
            "MlpPolicy",  # You can also use "CnnPolicy" for CNN architectures
            env=env,
            verbose=1,
            tensorboard_log=f"/ext_hdd/jjlee/jumpstart-rl/logs/{env_name}_guide_{model_name}",
        )
    
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
        callback=[WandbCallback(
            gradient_save_freq=10000,
            model_save_path=f"/ext_hdd/jjlee/jumpstart-rl/examples/models/{env_name}_guide_{model_name}",
            verbose=2,
        ),
                  EvalCallback(
            eval_env,
            n_eval_episodes=100,
            best_model_save_path=f"/ext_hdd/jjlee/jumpstart-rl/examples/models/{env_name}_guide_{model_name}"
        ),]
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
    args = parser.parse_args()
    main(args.env, args.timesteps, args.model)
