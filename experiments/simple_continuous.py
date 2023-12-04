# Write a simple continuous environment with single state and action range from -1 to 1

import gymnasium as gym
import numpy as np

class SimpleContinuous(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.state = np.array([0.0])
        self.goal = np.array([0.0])
        self.reward_range = (-1, 1)
        self.metadata = {}
        self.spec = None
        self.viewer = None
        self._max_episode_steps = 1000
        self._elapsed_steps = 0

    def step(self, action):
        self._elapsed_steps += 1
        reward = self.compute_reward(action)
        done = self._elapsed_steps >= self._max_episode_steps
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([0.0])
        self.goal = np.array([0.0])
        self._elapsed_steps = 0
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def compute_reward(self, action):
        # Compute the gaussian reward based on the action that has a mean of 0.3 and a standard deviation of 0.3
  
        reward = np.exp(-0.5 * ((action - 0.3) / 0.3) ** 2)
        # Bound the reward between -1 and 1
        reward = np.clip(reward, -1, 1)
        
        return reward
