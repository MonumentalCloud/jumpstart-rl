import sys
sys.path.append('/home/ubuntu/jjlee/jumpstart-rl/')


from experiments.gridworld import DiscreteGridworld, GoodOracleDiscrete, NeuralQLearningAgent, Buffer

import numpy as np
import random
from tqdm import tqdm

import wandb
from datetime import date

import torch

env = DiscreteGridworld(10, sparse=False, heatmap=True)
oracle = GoodOracleDiscrete(env)
agent = NeuralQLearningAgent(env)
training_iterations = range(10000)

buffer = Buffer()
horizons = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
horizon = len(horizons)-1

state = env.state
mean_reward = 0
mean_best_reward=0

def student_eval(env, agent, training_iterations, filename):

    state = env.reset(reset_heatmap=True)
    np_reward = np.zeros(len(training_iterations))
    np_dones = np.zeros(len(training_iterations))
    # run the training loop for the agent
    for i in training_iterations:
        cum_reward = 0
        for j in range(30):
            # Loop through training iterations online
            # Get the current state
            
            # Process the state data so that it returns the integer representation of where the agent is
            # find where the value occur in the 2d array of state and return the index
            # state = np.where(state == 1)
            # state = state[0][0] * env.grid_size + state[1][0]
            state = torch.tensor(state.flatten(), dtype=torch.float32)
            # Get the agent's action
            agent_action = agent.act(state) 
            
            # feed the action to the environment
            next_state, reward, done, truncate, info = env.step(agent_action)
            
            state = next_state
            cum_reward += reward
            
            if done:
                break
        np_reward[i] = cum_reward
        np_dones[i] = done
    # wandb.log({"mean_reward": np.mean(np_reward)})
    # wandb.log({"mean_dones": np.mean(np_dones)})
    
# wandb.init(project="jsrl",
#                 entity="jsrl-boys",
#             config={"gridworld": "True"},
#             group=f"{date.today()}",
#             sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
#             monitor_gym=True,
#             dir="/ext_hdd/jjlee/jumpstart-rl/logs/pointmaze_jsrl_curriculum", 
#         )
    
# for i in tqdm(training_iterations):
#     for j in range(30):
#         # Loop through training iterations online
#         # Get the current state
        
#         # Process the state data so that it returns the integer representation of where the agent is
#         # find where the value occur in the 2d array of state and return the index
#         # state = np.where(state == 1)
#         # state = state[0][0] * env.grid_size + state[1][0]
        
#         if j > horizons[horizon]:
#             # Get the agent's action
#             action = agent.act(state) 
#         else:
#             # Get the oracle's action 
#             action = oracle.predict(state)
            
#         # feed the action to the environment
#         next_state, reward, done, truncate, info = env.step(action)
        
        
#         # next_state_processed = np.where(next_state == 1)
#         # next_state_processed = next_state_processed[0][0] * env.grid_size + next_state_processed[1][0]
        
#         buffer.add([state, action, reward, next_state, done])
        
        
#         state = next_state
        
        
#         if done:
#             state = env.reset(reset_heatmap=False)
#             break
            
#     if i != 0 and i % 10 == 0:
#         samples = buffer.sample(10)
#         for sample in samples:
#             agent.learn(*sample)
            
#     if i % 1000 == 0:
#         # mean_reward = env.evaluate(agent, 10)
#         # print(mean_reward)
#         # print(horizons[horizon])
#         # print(mean_best_reward)
#         # First extract the q table from the agent's network
#         q_table = np.zeros((env.grid_size**2, env.action_space.n))
#         for j in range(env.grid_size**2):
#             # State is a 2d array size of grid x grid with agent's position marked as 1 and rest 0
#             state = np.zeros((env.grid_size, env.grid_size))
#             state[int(j/env.grid_size)][j%env.grid_size] = 1
#             state = torch.tensor(state.flatten(), dtype=torch.float32)
#             q_table[j] = agent.q.forward(state).detach().numpy()[0]
#         env.render(save=True, filename=f"arrow_map_{i}.png" , qmap=True, q_table=q_table, show=True)
#         # if mean_reward > mean_best_reward:
#         #     mean_best_reward = mean_reward
#         #     horizon -= 1
#         horizon -= 1
#             # print("Horizon decreased to {}".format(horizons[horizon]))
#         # else:
#             # print("Horizon kept at {}".format(horizons[horizon]))
#         state = env.reset(reset_heatmap=True)
#     # if i != 0 and i % (len(training_iterations)/10) == 0:
#     #     horizon -= 1
#         # env.render(save=True, filename=f"arrow_q_{i}" , qmap=True, q_table=agent.Q, show=True)

# Online Loop
for i in tqdm(training_iterations):
    for j in range(30):
        # Loop through training iterations online
        # Get the current state
        
        # Process the state data so that it returns the integer representation of where the agent is
        # find where the value occur in the 2d array of state and return the index
        # state = np.where(state == 1)
        # state = state[0][0] * env.grid_size + state[1][0]
        
        action = agent.act(state) 

            
        # feed the action to the environment
        next_state, reward, done, truncate, info = env.step(action)
        
        
        # next_state_processed = np.where(next_state == 1)
        # next_state_processed = next_state_processed[0][0] * env.grid_size + next_state_processed[1][0]
        
        buffer.add([state, action, reward, next_state, done])
        
        
        state = next_state
        
        
        if done:
            state = env.reset(reset_heatmap=False)
            break
            
    if i != 0 and i % 10 == 0:
        samples = buffer.sample(10)
        for sample in samples:
            agent.learn(*sample)
            
    if i % 1000 == 0:
        # mean_reward = env.evaluate(agent, 10)
        # print(mean_reward)
        # print(horizons[horizon])
        # print(mean_best_reward)
        # First extract the q table from the agent's network
        q_table = np.zeros((env.grid_size**2, env.action_space.n))
        for j in range(env.grid_size**2):
            # State is a 2d array size of grid x grid with agent's position marked as 1 and rest 0
            state = np.zeros((env.grid_size, env.grid_size))
            state[int(j/env.grid_size)][j%env.grid_size] = 1
            state = torch.tensor(state.flatten(), dtype=torch.float32)
            q_table[j] = agent.q.forward(state).detach().numpy()[0]
        env.render(save=True, filename=f"online_arrow_map_{i}.png" , qmap=True, q_table=q_table, show=True)
        #