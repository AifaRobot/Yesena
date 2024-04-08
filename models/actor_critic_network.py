import torch.nn as nn
import torch
import os
from activations import Swish

class ActorCriticNetwork(nn.Module):
    def __init__(self, in_chanels, n_actions):
        super(ActorCriticNetwork, self).__init__()

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
            nn.Linear(512, 256),
            Swish(),
            nn.Linear(256, n_actions), 
            nn.Softmax(dim=-1)
        )
        
        self.critic_output = nn.Sequential(            
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

        value = self.critic_output(new_hx)
        distribution = self.actor_output(new_hx)

        return distribution, value, new_hx

    def save(self, save_path):
        print('Guardando modelos...')
        
        if(os.path.exists(save_path) == False):
            os.mkdir(save_path)
        
        torch.save(self.state_dict(), save_path + '/actor_critic.pt')

    def load(self, save_path):
        
        if(os.path.isfile(save_path + '/actor_critic.pt')):
            print('Se ha cargado un modelo para la red neuronal')
            self.load_state_dict(torch.load(save_path + '/actor_critic.pt'))
        else:
            print('No se ha encontrado ningun podelo para la red neuronal')

    def get_new_hx(self):
        return torch.zeros(512)

class ActorCriticFactory():

    def __init__(self, in_chanels, n_actions):
        self.in_chanels = in_chanels
        self.n_actions = n_actions

    def create(self):
        return ActorCriticNetwork(self.in_chanels, self.n_actions)

