import torch.nn as nn
import torch
import os

class ActorCriticNetwork3(nn.Module):
    def __init__(self, in_chanels, n_actions):
        super(ActorCriticNetwork3, self).__init__()

        self.size_output_layer = 288

        self.network_input = nn.Sequential(
            nn.Linear(in_chanels, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        )
        
        self.actor_output = nn.Sequential(
            nn.Linear(128, 128), 
            nn.LeakyReLU(),
            nn.Linear(128, n_actions), 
            nn.Softmax(dim=-1)
        )
        
        self.critic_output = nn.Sequential(            
            nn.Linear(128, 128), 
            nn.LeakyReLU(),
            nn.Linear(128, 1), 
        )

    def forward(self, observation, _):
        network_output = self.network_input(observation)

        value = self.critic_output(network_output)
        distribution = self.actor_output(network_output)

        return distribution, value, torch.zeros(512)

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
            print('No se ha encontrado ningun modelo para la red neuronal')

    def get_new_hx(self):
        return torch.zeros(512)

class ActorCriticFactory3():

    def __init__(self, in_chanels, n_actions):
        self.in_chanels = in_chanels
        self.n_actions = n_actions

    def create(self):
        return ActorCriticNetwork3(self.in_chanels, self.n_actions)

