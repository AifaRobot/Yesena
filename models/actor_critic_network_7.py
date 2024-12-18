import torch.nn as nn
from activations import Swish
from models.network_base import NetworkBase

class ActorCriticNetwork7(NetworkBase):
    def __init__(self, in_chanels, n_actions):
        super(ActorCriticNetwork7, self).__init__('actor_critic', 128)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_chanels, 32, 3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            Swish(),
            nn.Flatten(1, 3),
        )

        self.actor_output = nn.Sequential(
            nn.Linear(128, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, n_actions), 
            nn.Softmax(dim=-1)
        )

        self.critic_output = nn.Sequential(            
            nn.Linear(128, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 1), 
        )

    def forward(self, observation, hx):
        network_output = self.encoder(observation)

        value = self.critic_output(network_output)
        distribution = self.actor_output(network_output).squeeze(0)

        return distribution, value, hx

class ActorCriticFactory7():

    def __init__(self, in_chanels, n_actions):
        self.in_chanels = in_chanels
        self.n_actions = n_actions

    def create(self):
        return ActorCriticNetwork7(self.in_chanels, self.n_actions)

