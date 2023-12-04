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
        
        return action, {}
import gym
from gym import spaces
import numpy as np
from torch import nn
import torch

# Define a good oracle that takes environment as an input and figures out the straight line path to the goal state
class GoodOracleDiscrete:
    def __init__(self, env):
        self.env = env
        # Initialize the gridworld size
        self.grid_size = env.grid_size
        
        # Initialize the goal state
        self.goal_state = env.goal_state
        
        # Initialize the action space
        self.action_space = env.action_space
        
    def predict(self, obs, *args, **kwargs):
        # Define the oracle policy that starts from the bottom-left corner and moves towards the goal state
        # First the agent calculates the angle between the agent's position and the goal state
        # Then the agent moves towards the goal state
        # Tabular policy that maps state to action
        # First divide the observation in int format into x and y coordinates by dividing by the grid size
        x = self.env.agent_position[0]
        y = self.env.agent_position[1]
        
        
        # Calculate the angle between the agent's position and the goal state
        
        
        angle = math.atan2(self.goal_state[1] - y, self.goal_state[0] - x)
        
        # Convert the angle to degrees
        angle = math.degrees(angle)
                
        # Calculate the action based on the angle
        # If the angle is between 0 and 45 degrees, move down
        if angle >= -45 and angle < 45:
            action = 1
        # If the angle is between 45 and 135 degrees, move right
        elif angle >= 45 and angle < 135:
            action = 3
        # If the angle is between 135 and 225 degrees, move up
        elif angle >= 135 and angle < 225:
            action = 0
        # If the angle is between 225 and 315 degrees, move left
        elif angle >= 225 and angle < 315:
            action = 2
        # If the angle is between 315 and 360 degrees, move down
        elif angle >= 315 or angle < -45:
            action = 1
            
        return action
    
    def reset(self, *args, **kwargs):
        pass
    
    def learn(self, *args, **kwargs):
        pass
    
    def save(self, *args, **kwargs):
        pass
    
    def load(self, *args, **kwargs):
        pass

# Define the class of simple Actor critic agent that learns a neural network policy and q function
class NeuralActorCriticAgent:
    def __init__(self, env, alpha=0.001, gamma=0.99, epsilon=0.5):
        # Initialize the learning rate
        self.alpha = alpha
        
        # Initialize the discount factor
        self.gamma = gamma
        
        # Initialize the exploration rate
        self.epsilon = epsilon
        
        # Initialize the action space
        self.action_space = env.action_space
        
        # Initialize the observation space
        self.observation_space = env.observation_space
        
        # Initialize the policy network
        self.policy = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 128, dtype=torch.float),
            nn.ReLU(),
            nn.Linear(128, 128, dtype=torch.float),
            nn.ReLU(),
            nn.Linear(128, 2, dtype=torch.float),
        )

        # Initialize the q network
        self.q = nn.Sequential(
            nn.Linear(self.observation_space.shape[0] + self.action_space.n, 128, dtype=torch.float),
            nn.ReLU(),
            nn.Linear(128, 128, dtype=torch.float),
            nn.ReLU(),
            nn.Linear(128, 1, dtype=torch.float),
        )

        # Initialize the optimizer for the policy network
        self.policy.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.alpha)
        self.policy.to(torch.float)
        
        # Initialize the optimizer for the q network
        self.q.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.alpha)
        self.q.to(torch.float)
        
        
    
    def act(self, state):
        # Select the greedy action
        mean, var = self.policy.forward(state).detach().numpy()
        
        # Ensure that the variance is non-negative
        var = max(var, 0.0001)
        
        # Sample an action from the normal distribution
        action = np.random.normal(mean, var)
        
        return action, (mean, var)
    
    def learn(self, state, action, reward, next_state, done):
        # Calculate the target for the Q network
        next_action, parameters = self.policy.forward(torch.tensor(next_state).to(torch.float))
        if type(reward) == np.ndarray:
            reward = float(reward)
        target = reward + self.gamma * self.q.forward(torch.tensor([torch.tensor(next_state), torch.tensor(next_action)], dtype=torch.float).T).to(torch.float)[0]

        # Calculate the loss for the Q network
        loss = nn.MSELoss()(self.q.forward(torch.tensor([torch.tensor(state), torch.tensor([action])], dtype=torch.float).T), target)

        # Update the Q network
        self.q.optimizer.zero_grad()
        loss.backward()
        self.q.optimizer.step()
        
        # Calculate the log likelihood distribution of the action
        mean, var = self.policy.forward(torch.tensor(state).to(torch.float))
        log_likelihood = -((action - mean) ** 2) / (2 * var) - torch.log(torch.sqrt(2 * np.pi * var))


        # Calculate the loss for the policy network which is advantage * log(pi)
        policy_loss = -log_likelihood * self.q.forward(torch.tensor([torch.tensor(state), torch.tensor([action])], dtype=torch.float).T)[0]

        # Update the policy network
        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step()
        
        return loss, policy_loss
        
# Define a class of simple Q learning agent that learns a neural network q function and greedy policy
class NeuralQLearningAgent:
    def __init__(self, env, alpha=0.001, gamma=0.99, epsilon=0.1):
        # Initialize the learning rate
        self.alpha = alpha
        
        # Initialize the discount factor
        self.gamma = gamma
        
        # Initialize the exploration rate
        self.epsilon = epsilon
        
        # Initialize the action space
        self.action_space = env.action_space
        
        # Initialize the observation space
        self.observation_space = env.observation_space
        
        # Initialize the q network
        self.q = nn.Sequential(
            nn.Linear(self.observation_space.n, 128, dtype=torch.float),
            nn.ReLU(),
            nn.Linear(128, 128, dtype=torch.float),
            nn.ReLU(),
            nn.Linear(128, self.action_space.n, dtype=torch.float),
            nn.Sigmoid()  # Apply sigmoid activation function
        )
        
        # Initialize the optimizer for the q network
        self.q.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.alpha)
        self.q.to(torch.float)
        
    def act(self, state):
        # Select the action based on the epsilon-greedy policy
        if np.random.random() < self.epsilon:
            # Select a random action
            action = self.action_space.sample()
        else:
            # Select the greedy action
            state = torch.tensor(state.flatten()).to(torch.float)
            action = np.argmax(self.q.forward(state).detach().numpy())
            
        return action
    
    def learn(self, state, action, reward, next_state, done):
        # Calculate the target for the Q network
        next_action = self.act(torch.tensor(next_state).to(torch.float))

        target = reward + self.gamma * self.q.forward(torch.tensor(next_state.flatten()).to(torch.float))[next_action]
        
        # Calculate the loss for the Q network
        loss = nn.MSELoss()(self.q.forward(torch.tensor(state.flatten()).to(torch.float)), target)
        
        # Update the Q network
        self.q.optimizer.zero_grad()
        loss.backward()
        self.q.optimizer.step()
        
        
        
        

# Define the class of simple Q learning agent that learns a tabular policy
class TabularQLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        # Randomly initialize the Q table
        self.Q = np.random.random((env.observation_space.n, env.action_space.n))
                
        # Initialize the learning rate
        self.alpha = alpha
        
        # Initialize the discount factor
        self.gamma = gamma
        
        # Initialize the exploration rate
        self.epsilon = epsilon
        
        # Initialize the action space
        self.action_space = env.action_space
        
    def act(self, state):
        # Select the action based on the epsilon-greedy policy
        if np.random.random() < self.epsilon:
            # Select a random action
            action = self.action_space.sample()
        else:
            # Select the greedy action
            action = np.argmax(self.Q[state])
            
        return action
    
    def learn(self, state, action, reward, next_state, done):
        # Update the Q table based on the Bellman equation
        self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])

# Define a simple Buffer for tabular Q learning
class Buffer:
    def __init__(self, buffer_size=1000):
        # Initialize the buffer size
        self.buffer_size = buffer_size
        
        # Initialize the buffer
        self.buffer = []
        
    def add(self, experience):
        # Add the experience to the buffer
        self.buffer.append(experience)
        
        # If the buffer size is exceeded, remove the oldest experience
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
    def sample(self, batch_size):
        # Sample a batch of experiences from the buffer
        batch = []
        if len(self.buffer) < batch_size:
            batch = random.sample(self.buffer, len(self.buffer))
        else:
            batch = random.sample(self.buffer, batch_size)
            
        # Return the batch
        return batch
    
    def __len__(self):
        # Return the length of the buffer
        return len(self.buffer)
    

class DiscreteGridworld(gym.Env):
    def __init__(self, grid_size=5, sparse=False, heatmap=False, seed=None):
        self.grid_size = grid_size

        # Define the discrete action space (up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Define the observation space
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)

        # Define the goal state
        self.goal_state = [grid_size - 1, grid_size - 1]

        # Define the initial state
        self.state = np.zeros((self.grid_size, self.grid_size))
        self.state[0, 0] = 1
        
        self.agent_position = [0, 0]
        
        self.sparse = sparse
        
        # Define the heatmap flag
        self.heatmap = heatmap
        if self.heatmap:
            self.heatmap_array = np.zeros((self.grid_size, self.grid_size))
            self.heatmap_array[0, 0] = 1
            
    def step(self, action):
        # Update the state based on the action
        if action == 0:
            # Move up
            self.agent_position[0] -= 1
        elif action == 1:
            # Move down
            self.agent_position[0] += 1
        elif action == 2:
            # Move left
            self.agent_position[1] -= 1
        elif action == 3:
            # Move right
            self.agent_position[1] += 1
        
        # Clip the state to make sure it's within the gridworld
        self.agent_position = np.clip(self.agent_position, 0, self.grid_size - 1)
            
        self.state = np.zeros((self.grid_size, self.grid_size))
        self.state[self.agent_position[0], self.agent_position[1]] = 1
        
        # Calculate the reward
        reward = self._calculate_reward()
        
        # Check if the episode is done
        done = self._is_done()
        
        if self.heatmap:
            self.heatmap_array[self.agent_position[0], self.agent_position[1]] += 1
        
        # Return the next state, reward, done, truncate, and info
        return self.state, reward, done, self.agent_position == self.goal_state, {}

    def reset(self, reset_heatmap=False, *args, **kwargs):
        # Reset the state to the initial state
        self.state = np.zeros((self.grid_size, self.grid_size))
        if self.heatmap and reset_heatmap:
            self.heatmap_array = np.zeros((self.grid_size, self.grid_size))
            self.heatmap_array[0, 0] = 1
        self.state[0, 0] = 1
        self.agent_position = [0, 0]
        return self.state

    def render(self, mode='human', heatmap=False, qmap=False, q_table=None, save=False, filename=None, show=True, *args, **kwargs):
        # Using the matplotlib library to plot the gridworld
        import matplotlib.pyplot as plt
        
        if qmap:
            assert q_table is not None, "Please specify a q_table"
        if save:
            assert filename is not None, "Please specify a filename"
            
        # Clear everything from the plt
        plt.clf()
        
        # # Plot the gridworld
        # plt.imshow(self.state, cmap='white', vmin=0, vmax=1)
        
        # Draw a grid dividing up the space
        plt.grid(True, which='both', color='lightgray', linewidth=0.5)
        
        # Set the x and y ticks
        plt.xticks(np.arange(-.5, self.grid_size, 1))
        plt.yticks(np.arange(-.5, self.grid_size, 1))
        
        # Plot the starting state in orange
        plt.scatter(0, 0, c='orange')
        
        # Plot the goal state in green
        plt.scatter(self.goal_state[0], self.goal_state[1], c='green')
        
        # Plot the agent's q values as a heatmap of arrows
        if qmap:
            # Define the arrows
            arrows = ['↓', '↑', '←', '→']
            
            # Define the x and y coordinates
            x = np.arange(0, self.grid_size, 1)
            y = np.arange(0, self.grid_size, 1)
            
            # Define the meshgrid
            X, Y = np.meshgrid(x, y)
                    
            
            # Plot the arrows in colors corresponding to the q values so that the arrows are darker when the q values are higher
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    state = i * self.grid_size + j
                    # plot four arrows in each state
                    for action in range(4):
                        # Define the q value from the q table which is one dimensional flattened array
                        q_value = q_table[i * self.grid_size + j][action] + 1

                        
                        # Calculate the color based on the q value
                        color = 1 - ((q_value+1) / (np.max(q_table)+2))
                        
                        # Plot the arrow slightly off center so they don't overlap
                        if action == 1:
                            plt.text(j + 0.1, i + 0.3, arrows[action], color=str(color))
                        elif action == 0:
                            plt.text(j + 0.1, i - 0.1, arrows[action], color=str(color))
                        elif action == 2:
                            plt.text(j - 0.1, i + 0.1, arrows[action], color=str(color))
                        elif action == 3:
                            plt.text(j + 0.3, i + 0.1, arrows[action], color=str(color))
                         
                        # Get rid of axis
                        plt.axis('off')
                        
                        
                        
                        
        # Plot the heatmap
        if self.heatmap and heatmap:
            plt.imshow(self.heatmap_array, cmap='hot', interpolation='nearest', alpha=0.5)
            # Overlay the number of times each state has been visited
            # for i in range(self.grid_size):
            #     for j in range(self.grid_size):
            #         plt.text(j, i, str(self.heatmap_array[i, j]), horizontalalignment='center', verticalalignment='center')
        
        if save:
            # Save the plot
            plt.savefig(filename)
        
        if show:
            # Show the plot
            plt.show()
        
    def evaluate(self, policy, num_episodes=100):
        # Define a list to store the rewards
        rewards = []
        
        # Iterate through each episode
        for episode in range(num_episodes):
            # Reset the state
            state = self.reset()
            
            # Initialize the episode reward
            episode_reward = 0
            
            # Iterate until the episode is done
            done = False
            for step in range(300):
                state = torch.tensor(state.flatten()).to(torch.float)
                # Select an action based on the policy
                action = policy.act(state)
                
                # Take a step
                next_state, reward, done, _, _ = self.step(action)
                
                # Update the episode reward
                episode_reward += reward
                
                # Update the state
                state = next_state
                
            # Append the episode reward to the list of rewards
            rewards.append(episode_reward)
            
        # Return the average reward
        return np.mean(rewards)
        
    def _calculate_reward(self):
        if self.sparse:
            # Calculate the reward based on the current state
            if np.array_equal(self.agent_position, self.goal_state):
                reward = 1
            else:
                reward = 0
            return reward
        # Calculate the Euclidean distance between the agent_position and the goal_state
        distance = np.linalg.norm(self.agent_position - self.goal_state)
        
        # Apply a non-linear transformation to the distance to make the reward increase disproportionately
        reward = 1 / (distance + 1)
        
        return reward
    
    def _is_done(self):
        # Check if the episode is done (e.g., reaching a terminal state)
        return np.array_equal(self.agent_position, self.goal_state)
    
    

class ContinuousGridworld(gym.Env):
    def __init__(self, seed):
        # Define the gridworld size
        self.grid_size = 32

        # Define the discrete 4 action space (up, down, left, right)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.int)

        # Define the observation space
        self.observation_space = Box(low=0, high=1, shape=(2,), dtype=np.int)
        
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
            if path == 1:
                color = 'red'
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
        


        