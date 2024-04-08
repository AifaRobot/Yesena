import torch.nn as nn
import torch
import os
from activations import Swish

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, in_chanels, size_output_layer):
        super(ActorNetwork, self).__init__()

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
            nn.Linear(256, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, observation):
        network_output = self.encoder(observation)

        if(network_output.size(dim=0) == 1):
            network_output = network_output.view(self.size_output_layer)

        distribution = self.output(network_output)

        return distribution

class CriticNetwork(nn.Module):
    def __init__(self, in_chanels, size_output_layer):
        super(CriticNetwork, self).__init__()

        self.size_output_layer = size_output_layer
       
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chanels, 32, 3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            Swish(),
            nn.Flatten(),
        )
        
        self.output = nn.Sequential(
            nn.Linear(512, 256),
            Swish(),
            nn.Linear(256, 1),
        )
        
        self.gru = nn.GRUCell(self.size_output_layer, 512)

    def forward(self, observation, hx):
        network_output = self.encoder(observation)

        if(network_output.size(dim=0) == 1):
            network_output = network_output.view(self.size_output_layer)

        new_hx = self.gru(network_output, (hx))

        value = self.output(new_hx)

        return value, new_hx.detach()

class ActorCriticNetwork2():
    def __init__(self, in_chanels, n_actions):
        self.size_output_layer = 288

        self.actor = ActorNetwork(n_actions, in_chanels, self.size_output_layer)
        self.critic = CriticNetwork(in_chanels, self.size_output_layer)

    def forward(self, observation, hx):
        value, hx = self.critic(observation, hx)
        distribution = self.actor(observation)

        return distribution, value, hx

    def get_new_hx(self):
        return torch.zeros(512)

    def save(self, save_path):
        print('Guardando modelos...')
        
        if(os.path.exists(save_path) == False):
            os.mkdir(save_path)
        
        torch.save(self.actor.state_dict(), save_path + '/actor.pt')
        torch.save(self.critic.state_dict(), save_path + '/critic.pt')

    def load(self, save_path):
        if(os.path.isfile(save_path + '/actor.pt')):
            print('Se ha cargado un modelo para la red neuronal actor')
            self.actor.load_state_dict(torch.load(save_path + '/actor.pt'))
        else:
            print('No se ha encontrado ningun podelo para la red neuronal')

        if(os.path.isfile(save_path + '/critic.pt')):
            print('Se ha cargado un modelo para la red neuronal critico')
            self.critic.load_state_dict(torch.load(save_path + '/critic.pt'))
        else:
            print('No se ha encontrado ningun modelo de curiosidad para la red neuronal')

class ActorCriticFactory2():
    def __init__(self, in_chanels, n_actions):
        self.in_chanels = in_chanels
        self.n_actions = n_actions

    def create(self):
        return ActorCriticNetwork2(self.in_chanels, self.n_actions)