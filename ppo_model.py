import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
import numpy as np

def layer_init(layer,std = np.sqrt(2),bias_c = 0.0):
    nn.init.orthogonal_(layer.weight,std)
    nn.init.constant_(layer.bias,bias_c)
    return layer

class ppo_net(nn.Module):
    def __init__(self,obs_space_channel,action_space_out):
        super().__init__()
        self.main = nn.Sequential(
            layer_init(nn.Conv2d(obs_space_channel[0],32,kernel_size=8,stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32,64,kernel_size=4,stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64,64,kernel_size=3,stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64*7*7,512)),
            nn.ReLU()
        )

        self.policy = nn.Sequential(
            layer_init(nn.Linear(512,action_space_out),std=0.01)
        )
        self.value = nn.Sequential(
            layer_init(nn.Linear(512,1),std=1)
        )

    def forward(self,x,action=None):
        x = self.main(x)
        probs = Categorical(logits=self.policy(x))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.value(x)
    
    def forward_value(self, x):
        x = self.main(x)
        return self.value(x)
    
    def select_action(self, x):
        x = self.main(x)
        probs = Categorical(logits=self.policy(x))
        return probs.sample()









