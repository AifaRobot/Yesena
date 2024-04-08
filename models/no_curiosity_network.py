import torch

class NoCuriosityNetwork():
    def __init__(self):
        super(NoCuriosityNetwork, self).__init__()


    def calc_reward(self, observation, next_observation, action):
        return 0

    def load(self, save_path):
        pass

    def save(self, save_path):
        pass

    def load_state_dict(self, _):
        pass

    def share_memory(self):
        pass

    def parameters(self):
        return torch.tensor([])
    
    def state_dict(self):
        return torch.tensor([])

    def calc_loss(self, observations, next_observations, actions):

        return torch.tensor(0).detach()

class NoCuriosityFactory():

    def create(self):
        return  NoCuriosityNetwork()

