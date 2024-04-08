import torch.nn as nn
import torch
import os
from activations import Swish

class CuriosityNetwork(nn.Module):
    def __init__(self, in_chanels, n_actions, alpha, beta):
        super(CuriosityNetwork, self).__init__()
        
        self.size_output_layer = 288
        self.alpha = alpha
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Conv2d(in_chanels, 32, 3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            Swish(),
            nn.Flatten()
        )
        
        self.inverse_model = nn.Sequential(
            nn.Linear(self.size_output_layer*2, 512),
            Swish(),
            nn.Linear(512, n_actions)
        )
        
        self.forward_model = nn.Sequential(
            nn.Linear(self.size_output_layer+1, 512),
            Swish(),
            nn.Linear(512, self.size_output_layer)
        )
    
        self.inverse_loss = nn.CrossEntropyLoss()
        self.forward_loss = nn.MSELoss()


    def forward(self, observation, next_observation, action):
        phi = self.encoder(observation)
        phi_new = self.encoder(next_observation)

        pi_logits = self.inverse_model(torch.cat([phi, phi_new], dim=1))
        phi_hat_new = self.forward_model(torch.cat([phi, action], dim=1))

        return phi_new, phi_hat_new, pi_logits


    def calc_reward(self, observation, next_observation, action):
        phi_new, phi_hat_new, _ = self.forward(observation, next_observation, action.view(-1, 1))

        intrinsic_reward = self.alpha*0.5*((phi_hat_new-phi_new).pow(2)).mean(dim=1)

        return intrinsic_reward.item()

    def calc_loss(self, observations, next_observations, actions):
        phi_new, phi_hat_new, pi_logits = self.forward(observations, next_observations, actions.view(-1, 1))

        L_I = (1 - self.beta) * self.inverse_loss(pi_logits, actions)
        
        L_F = self.beta * self.forward_loss(phi_hat_new, phi_new)

        return L_I + L_F

    def save(self, save_path):
        print('Guardando modelos...')
        
        if(os.path.exists(save_path) == False):
            os.mkdir(save_path)
        
        torch.save(self.state_dict(), save_path + '/curiosity.pt')

    def load(self, save_path):
        
        if(os.path.isfile(save_path + '/curiosity.pt')):
            print('Se ha cargado un modelo para la red neuronal Curiosidad')
            self.load_state_dict(torch.load(save_path + '/curiosity.pt'))
        else:
            print('No se ha encontrado ningun podelo para la red neuronal')


class CuriosityFactory():

    def __init__(self, in_channels, n_actions, alpha, beta):
        self.in_channels = in_channels
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta

    def create(self):
        return  CuriosityNetwork(self.in_channels, self.n_actions, self.alpha, self.beta)

