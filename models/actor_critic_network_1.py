import torch.nn as nn
from activations import Swish
from models.network_base import NetworkBase

class ActorCriticNetwork1(NetworkBase):
    def __init__(self, in_chanels, n_actions):
        super(ActorCriticNetwork1, self).__init__('actor_critic', 128)

        self.size_output_layer = 288

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
            #nn.Linear(512, 256),
            nn.Linear(128, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, n_actions), 
            nn.Softmax(dim=-1)
        )

        self.critic_output = nn.Sequential(            
            #nn.Linear(512, 256),
            nn.Linear(128, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 1), 
        )

        self.gru = nn.GRUCell(self.size_output_layer, 128)

    def forward(self, observation, hx):
        network_output = self.encoder(observation)
        
        if(network_output.size(dim=0) == 1):
            network_output = network_output.view(self.size_output_layer)

        new_hx = self.gru(network_output, (hx))
    
        value = self.critic_output(new_hx)
        distribution = self.actor_output(new_hx)

        return distribution, value, new_hx.detach()

class ActorCriticFactory1():

    def __init__(self, in_chanels, n_actions):
        self.in_chanels = in_chanels
        self.n_actions = n_actions

    def create(self):
        return ActorCriticNetwork1(self.in_chanels, self.n_actions)

