from rewards import Reward
from rewards.utils import discount

class Advantage(Reward):
    def __init__(self, gamma, tau):
        self.gamma = gamma
        self.tau = tau

    def calculate_generalized_advantage_estimate(self, rewards, values, dones):
        delta_t = rewards + self.gamma * values[1:] * dones - values[:-1]

        return discount(delta_t, dones, self.gamma*self.tau)
