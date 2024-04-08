from rewards.utils import discount

class Reward:
    def __init__(self, gamma):
        self.gamma = gamma

    def calculate_discounted_rewards(self, rewards, dones):
        return discount(rewards, dones, self.gamma)

