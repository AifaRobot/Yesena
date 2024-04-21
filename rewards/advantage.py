import torch
from rewards import Reward
from rewards.utils import discount

class Advantage(Reward):
    def __init__(self, gamma, tau, normalize = False):
        self.gamma = gamma
        self.tau = tau
        self.normalize = normalize

    def calculate_generalized_advantage_estimate(self, rewards, values, dones):
        delta_t = rewards + self.gamma * values[1:] * dones - values[:-1]

        advantages = discount(delta_t, dones, self.gamma*self.tau)

        if self.normalize:
            advantages = (advantages - torch.mean(advantages) ) / ( torch.std(advantages) + 1e-8)

        return advantages
