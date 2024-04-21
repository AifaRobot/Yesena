import gym
from agents import Agent
from wrappers.gym_env import GymEnvCartpole, GymEnvAcrobot

class GymTestAcrobot():

    def __init__(self, env_name, in_channels, batch_size, render):
        self.env_name = env_name
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.render = render

        raw_env = gym.make(self.env_name, render_mode="human")
        self.worker = GymEnvAcrobot(raw_env)

    def run(self, method):   
        agente = Agent(
            method.main_model_factory,
            method.curiosity_model_factory,
            method.save_path
        )
        
        while True:
            done = False
            observation = self.worker.reset()

            hx = agente.get_new_hx()
            
            while not done: 
                action, next_hx = agente.get_action_max_prob(observation, hx)

                next_state, _, done = self.worker.step(action.item())

                observation = next_state
                hx = next_hx


class GymTestCartpole():

    def __init__(self, env_name, in_channels, batch_size, render):
        self.env_name = env_name
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.render = render

        raw_env = gym.make(self.env_name, render_mode="human")
        self.worker = GymEnvCartpole(raw_env)

    def run(self, method):   
        
        agente = Agent(
            method.main_model_factory,
            method.curiosity_model_factory,
            method.save_path
        )

        while True:
            done = False
            observation = self.worker.reset()

            hx = agente.get_new_hx()
            while not done: 
                action, next_hx = agente.get_action_max_prob(observation, hx)

                next_state, _, done = self.worker.step(action.item())

                observation = next_state
                hx = next_hx
