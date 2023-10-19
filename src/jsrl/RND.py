import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

# Import OrderedDict type
from collections import OrderedDict

class RNDModel(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_size=256):
        super(RNDModel, self).__init__()
        self.target_net = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_shape)
        )
        self.predictor_net = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_shape)
        )
        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=1e-3)

    def forward(self, x):
        target_out = self.target_net(x)
        predict_out = self.predictor_net(x)
        return target_out, predict_out

    def update(self, obs):
        obs = torch.FloatTensor(obs)
        target_out, predict_out = self(obs)
        target_out.detach_()
        loss = (target_out - predict_out).pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        wandb.log({"loss": loss})
        self.optimizer.step()

class RNDEstimator:
    def __init__(self, input_shape, output_shape, hidden_size=256, buffer_size=10000, batch_size=32):
        self.model = RNDModel(input_shape, output_shape, hidden_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.obs_buffer = np.zeros((buffer_size, input_shape))
        self.obs_ptr = 0
        self.obs_count = 0

    def update_obs(self, obs):
        self.obs_buffer[self.obs_ptr] = obs
        self.obs_ptr = (self.obs_ptr + 1) % self.buffer_size
        self.obs_count = min(self.obs_count + 1, self.buffer_size)

    def get_reward(self, obs):
        # obs = torch.FloatTensor(obs)
        # convert the dictionary values into a tensor
        if type(obs) == OrderedDict:
            obs = torch.FloatTensor(list(obs['observation']))
        _, predict_out = self.model(obs)
        target_out = self.model.target_net(obs).detach()
        reward = (target_out - predict_out).pow(2).sum(dim=1).sqrt()
        return reward.detach().numpy()

    def update_model(self):
        obs_indices = np.random.choice(self.obs_count, self.batch_size, replace=True)
        obs_batch = self.obs_buffer[obs_indices]
        self.model.update(obs_batch)