from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.core import Wrapper
from gym.core import Env



import numpy as np
import sys
sys.path.append('/home/ubuntu/jjlee/jumpstart-rl/')
from experiments.gridworld import ContinuousGridworld

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
            truncated = info["success"] or truncated
            
        return obs, reward, terminated, truncated, info
    
    
    def reset(self, seed=None):
        return self.env.reset(seed=seed)

# Create the ContinuousGridworld environment
env = ContinuousGridworld(seed=0)
env = MetaWorldWrapper(env, sparse=True, truncate_when_done=False)

# Wrap the environment with DummyVecEnv
env = DummyVecEnv([lambda: env])

# Create an instance of the SAC algorithm
model = SAC("MlpPolicy", env)

pure_env = ContinuousGridworld(seed=0)

# Train the SAC algorithm on the environment
model.learn(total_timesteps=int(1e5))

def extract_path(policy):
    x_path = []
    y_path = []
    state,_ = pure_env.reset()
    timestep = 0
    while True and timestep < 100:
        action, _ = policy.predict(state)
        state, _, done, _, _ = pure_env.step(action)
        x_path.append(state[0])
        y_path.append(state[1])
        timestep += 1
        if done:
            timestep = 0
            break
    return x_path, y_path

student_policy = model.policy
x_student, y_student = extract_path(student_policy)

paths = {
    "x": [x_student],
    "y": [y_student],
}

env.envs[0].render_path(paths)


