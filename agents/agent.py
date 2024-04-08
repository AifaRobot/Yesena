import torch
from torch.distributions import Categorical

class Agent():
    def __init__(self, main_model_factory, curiosity_model_factory, save_path):

        self.actor_critic = main_model_factory.create()
        self.curiosity = curiosity_model_factory.create()

        self.save_path = save_path

        self.load_models()

    def save_models(self):
        self.actor_critic.save(self.save_path)
        self.curiosity.save(self.save_path)

    def load_models(self):
        self.actor_critic.load(self.save_path)
        self.curiosity.load(self.save_path)

    def get_action(self, observation, hx = None):
        distribution, value, next_hx = self.actor_critic.forward(observation, hx)

        m = Categorical(distribution)
        action = m.sample()
        log_prob = m.log_prob(action)

        return log_prob, value.squeeze(0), action, next_hx, distribution

    def get_action_max_prob(self, observation, hx):
        distribution, _, next_hx = self.actor_critic.forward(observation, hx)

        action = torch.argmax(distribution)

        return action, next_hx

    def get_new_hx(self):
        return self.actor_critic.get_new_hx()