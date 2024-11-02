import torch
import torch.nn as nn
import os

class NetworkBase(nn.Module):
    def __init__(self, file_name, hidden_state = 512):
        super(NetworkBase, self).__init__()

        self.file_name = file_name
        self.hidden_state = hidden_state
    
    def save(self, save_path):
        print('Guardando modelo...')
        
        if(os.path.exists(save_path) == False):
            os.mkdir(save_path)
        
        torch.save(self.state_dict(), save_path + '/' + self.file_name + '.pt')

    def load(self, save_path):
        
        if(os.path.isfile(save_path + '/' + self.file_name + '.pt')):
            print('Se ha cargado un modelo para la red neuronal')
            self.load_state_dict(torch.load(save_path + '/' + self.file_name + '.pt'))
        else:
            print('No se ha encontrado ningun modelo para la red neuronal')

    def get_new_hx(self):
        return torch.zeros(self.hidden_state)
