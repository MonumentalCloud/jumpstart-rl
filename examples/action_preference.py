# Visualization code for action preference of a policy in a PointMaze environment
#

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.base_class import BaseAlgorithm
import argparse
import os
import wandb
from wandb.integration.sb3 import WandbCallback

def main(env_name):
    env = gym.make(env_name, max_episode_steps=150, render_mode="rgb_array") #continuing_task=False, max_episode_steps=150)
    policy = TD3.load(f"/home/ubuntu/jjlee/jumpstart-rl/examples/models/{env_name}_TD3/best_model.zip").policy
    
    # Now we can use the trained agent to perform action preference
    obs, _ = env.reset()
  
    # get the action preference of the policy by querying the policy for every x,y coordinate in the maze
    # Initialize the action preference matrix so that it is a 2D array of size (x, y, 2) of the maze
    maze = env.unwrapped.maze
    action_preference = np.zeros((maze.map_width, maze.map_length, 2))
    # Iterate through every x,y coordinate in the maze
    for i in range(action_preference.shape[0]):
        for j in range(action_preference.shape[1]):
            if maze.maze_map[j][i] == 0:
                # Set the observation to the current x,y coordinate
                obs["observation"][0] = i #- (maze.map_width / 2)
                obs["observation"][1] = j #- (maze.map_length / 2)
                obs["observation"][2] = 1
                obs["observation"][3] = 1
                obs["achieved_goal"][0] = i #- (maze.map_width / 2)
                obs["achieved_goal"][1] = j
                # Query the policy for the action preference
                policy.jsrl_evaluation=True
                action_preference[i,j] = policy.predict(obs,0, deterministic=True)[0]
    
    
    
    # Plot heatmap with trajectory and goal
    fig, ax = plt.subplots(figsize=(10,10), dpi=200)
    # Set the size of the figure as the size of the action_preference array

    # Set the background as the rendered maze
    ax.imshow(maze.maze_map, cmap="Blues", extent=[0, action_preference.shape[0], action_preference.shape[1],0], aspect="equal")
    
    # Plot the goal
    ax.scatter(obs['desired_goal'][0] + (maze.map_width / 2), obs['desired_goal'][1] + (maze.map_length/2), marker="*", s=100, color="red")

    # Plot the heatmap
    # ax.imshow(action_preference[:,:,0], alpha=0.5, cmap="bwr", vmin=-1, vmax=1, extent=[0, render.shape[1], render.shape[0], 0])
    # Plot the arrows every 10 coordinates
    for i in range(action_preference.shape[0]):
        for j in range(action_preference.shape[1]):
            if maze.maze_map[j][i] == 0:
                ax.arrow(i + 0.5, j + 0.5, action_preference[i,j,0]*0.3, action_preference[i,j,1]*0.3, head_width=0.2, head_length=0.3, length_includes_head=False, color="black")
    # Set the title
    ax.set_title(f"{env_name} Action Preference of Policy")
    # Save the figure
    fig.savefig(f"{env_name}_action_preference.png")
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PointMaze_UMaze-v3")
    parser.add_argument("--guide_steps", type=int, default=100000)
    args = parser.parse_args()
    main(args.env)