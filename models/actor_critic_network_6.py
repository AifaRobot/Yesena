import torch.nn as nn
from models.network_base import NetworkBase

class ActorNetwork(NetworkBase):
    def __init__(self, n_actions, in_chanels, size_output_layer):
        super(ActorNetwork, self).__init__('actor')

        self.size_output_layer = size_output_layer

        self.encoder = nn.Sequential(
            nn.Conv2d(in_chanels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.output = nn.Sequential(
            nn.Linear(288, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, observation):
        network_output = self.encoder(observation)

        if(network_output.size(dim=0) == 1):
            network_output = network_output.view(self.size_output_layer)

        distribution = self.output(network_output)
        
        return distribution

class CriticNetwork(NetworkBase):
    def __init__(self, in_chanels, size_output_layer):
        super(CriticNetwork, self).__init__('critic')

        self.size_output_layer = size_output_layer

        self.encoder = nn.Sequential(
            nn.Conv2d(in_chanels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.output = nn.Sequential(            
            nn.Linear(288, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, observation):
        network_output = self.encoder(observation)

        if(network_output.size(dim=0) == 1):
            network_output = network_output.view(self.size_output_layer)

        value = self.output(network_output)

        return value

class ActorCriticNetwork6():
    def __init__(self, in_chanels, n_actions):
        self.size_output_layer = 288

        self.actor = ActorNetwork(n_actions, in_chanels, self.size_output_layer)
        self.critic = CriticNetwork(in_chanels, self.size_output_layer)

    def forward(self, observation, _):
        value = self.critic(observation)
        distribution = self.actor(observation)

        new_hx = self.get_new_hx()

        return distribution, value, new_hx

    def save(self, save_path):
        self.actor.save(save_path)
        self.critic.save(save_path)

    def load(self, save_path):
        self.actor.load(save_path)
        self.critic.load(save_path)

    def get_new_hx(self):
        return self.critic.get_new_hx()

class ActorCriticFactory6():
    def __init__(self, in_chanels, n_actions):
        self.in_chanels = in_chanels
        self.n_actions = n_actions

    def create(self):
        return ActorCriticNetwork6(self.in_chanels, self.n_actions)