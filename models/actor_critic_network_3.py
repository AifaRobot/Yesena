import torch.nn as nn
from models.network_base import NetworkBase

class ActorCriticNetwork3(NetworkBase):
    def __init__(self, in_chanels, n_actions):
        super(ActorCriticNetwork3, self).__init__('actor_critic')

        self.size_output_layer = 288

        self.actor_output = nn.Sequential(
            nn.Linear(in_chanels, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        
        self.critic_output = nn.Sequential(            
            nn.Linear(in_chanels, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, observation, _):
        value = self.critic_output(observation)
        distribution = self.actor_output(observation)

        new_hx = self.get_new_hx()

        return distribution, value, new_hx

class ActorCriticFactory3():

    def __init__(self, in_chanels, n_actions):
        self.in_chanels = in_chanels
        self.n_actions = n_actions

    def create(self):
        return ActorCriticNetwork3(self.in_chanels, self.n_actions)

