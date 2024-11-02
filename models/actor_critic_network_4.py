import torch.nn as nn
from models.network_base import NetworkBase

class ActorCriticNetwork4(NetworkBase):
    def __init__(self, in_chanels, n_actions):
        super(ActorCriticNetwork4, self).__init__('actor_critic')

        self.size_output_layer = 288

        self.encoder = nn.Sequential(
            nn.Conv2d(in_chanels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(1, 3),
        )

        self.actor_output = nn.Sequential(
            nn.Linear(self.size_output_layer, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions), 
            nn.Softmax(dim=-1)
        )
        
        self.critic_output = nn.Sequential(            
            nn.Linear(self.size_output_layer, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1), 
        )

    def forward(self, observation, _):
        network_output = self.encoder(observation)

        value = self.critic_output(network_output)
        distribution = self.actor_output(network_output)

        new_hx = self.get_new_hx()

        return distribution.squeeze(0), value.squeeze(0), new_hx

class ActorCriticFactory4():

    def __init__(self, in_chanels, n_actions):
        self.in_chanels = in_chanels
        self.n_actions = n_actions

    def create(self):
        return ActorCriticNetwork4(self.in_chanels, self.n_actions)

