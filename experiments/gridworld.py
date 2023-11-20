import gym
from gymnasium.spaces.box import Box
import numpy as np
import math
import gym
from gym import spaces
import numpy as np
import math
import random
import matplotlib.pyplot as plt

class BadOracle:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        
    def predict(self, obs, *args, **kwargs):
        # Define the oracle policy that starts from the bottom-left corner and moves towards the goal state
        # First it moves towards the right, and once it hits the wall, it moves upwards
        # Tabular policy that maps state to action
        # Until the agent reaches the bottom-right corner, keep moving right
        try: 
            if obs[0][0] < self.grid_size:
                action = np.array([1, 0])
            # Once the agent reaches the bottom-right corner, keep moving upwards
            else:
                action = np.array([0, 1])
        except:
            if obs[0] < self.grid_size:
                action = np.array([1, 0])
            # Once the agent reaches the bottom-right corner, keep moving upwards
            else:
                action = np.array([0, 1])
        
            
        return action


class ContinuousGridworld(gym.Env):
    def __init__(self, seed):
        # Define the gridworld size
        self.grid_size = 10

        # Define the action space
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Define the observation space
        self.observation_space = Box(low=0, high=1, shape=(2,), dtype=np.float32)
        
        # Define the goal state
        self.goal_state = [10,10]

        # Define the initial state
        self.state = np.zeros((self.grid_size, self.grid_size))
        
        self.timestep = 0
        
        # Define the agent's position
        self.agent_position = [0, 0]
        
        self.bad_oracle = BadOracle(self.grid_size)
  
    def step(self, action):
        # Update the agent's position based on the action
        if action[1] == None:
            action = action[0]
        self.agent_position = [self.agent_position[0] + action[0], self.agent_position[1] + action[1]]
        # Clip the agent's position to make sure it's within the gridworld
        self.agent_position = np.clip(self.agent_position, 0, self.grid_size)

        # Calculate the reward
        reward = self._calculate_reward()
        
        # Check if the episode is done
        done = self._is_done()
        
        # Return the next state, reward, done, truncate, and info

        return self.agent_position, reward, done, self.timestep == 100, {"success": self._is_done()}

    def _calculate_reward(self):

        # Calculate the Euclidean distance between the agent_position and the goal_state
        distance = np.linalg.norm(self.agent_position - self.goal_state)

        # Apply a non-linear transformation to the distance to make the reward increase disproportionately
        reward = 1 / (distance + 1)

        return reward

    def _is_done(self):
        # Check if the episode is done (e.g., reaching a terminal state)
        return np.array_equal(self.agent_position, self.goal_state)
        
    def render(self):

        y,x = self.agent_position


        # using the plt library to plot the grid_map
        import matplotlib.pyplot as plt

        # plot self.state and draw a grid dividing up the space
        plt.imshow(self.state, cmap='gray', vmin=0, vmax=1)
        plt.grid(True, which='both', color='lightgrey', linewidth=0.5)
        plt.xticks(np.arange(-.5, self.grid_size, 1))
        plt.yticks(np.arange(-.5, self.grid_size, 1))
        

        

        # Plot the agent's position in orange
        plt.scatter(x, y, c='orange')
        
        # Plot the goal state in green
        plt.scatter(self.goal_state[0], self.goal_state[1], c='green')
        


        # Show the plot
        plt.show()
        
        
    def reset(self, seed=None, *args, **kwargs):
        # Set the seed value
        np.random.seed(seed)
        
        # Reset the state
        self.state = np.zeros((self.grid_size, self.grid_size))
        self.state[0, 0] = 1
        
        # Reset the timestep
        self.timestep = 0
        
        # Reset the agent's position
        self.agent_position = [0, 0]
        
        return self.agent_position, {}


    def render_path(self, paths):
        # paths dictionary is divided up into arrays of x and y coordinates
        # Paths can contain multiple paths, so in path['x'] the first array is the x coordinates of the first path and the second array is the x coordinates of the second path
        # The same goes for path['y']
        x = paths['x']
        y = paths['y']
        
        # Repeat the same procedure as render function but add the path and connect them together in random colors
        # plot self.state and draw a grid dividing up the space
        plt.imshow(self.state, cmap='gray', vmin=0, vmax=1)
        plt.grid(True, which='both', color='lightgrey', linewidth=0.5)
        plt.xticks(np.arange(-.5, self.grid_size, 1))
        plt.yticks(np.arange(-.5, self.grid_size, 1))
        
        # Plot the agent's positions and connect the dots using the line function
        for path in range(len(x)):
            color = (random.random(), random.random(), random.random())  # Generate a random RGB color
            for i in range(len(x[path])):
                plt.scatter(x[path][i], y[path][i], c=color)
                if i != len(x[path]) - 1:
                    plt.plot([x[path][i], x[path][i+1]], [y[path][i], y[path][i+1]], c=color)

        # Plot the goal state in green
        plt.scatter(self.goal_state[0], self.goal_state[1], c='green')
        
        # save the plot
        plt.savefig('path.png')
        
    def get_env_state(self):
        return self.state  
        


        