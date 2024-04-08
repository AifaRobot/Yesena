from wrappers.skip_and_frames_env import SkipAndFramesEnv
from wrappers.vizdoomenv_basic import Vizdoomenv
from wrappers.vizdoomenv_my_way_home import Vizdoomenvmywayhome

import torch
from agents import Agent

class VizdoomBasicTest():

    def __init__(self, env_name, in_channels, batch_size, render):
        self.env_name = env_name
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.render = render

    def create_worker(self):
        raw_env = Vizdoomenv(self.render)
        worker = SkipAndFramesEnv(raw_env, self.in_channels)

        return worker

    def run(self, method):
        worker = self.create_worker()

        agente = Agent(
            method.main_model_factory,
            method.curiosity_model_factory,
            method.save_path
        )
        
        while True:
            done = False
            observation = worker.reset()
            hx = torch.zeros(512)

            while not done: 
                action, next_hx = agente.get_action_max_prob(observation, hx)

                next_state, reward, done = worker.step(action.item())

                observation = next_state
                hx = next_hx

                if done:
                    print('Gano' if reward > 0 else 'Perdio')


class VizdoomMyWayHomeTest(VizdoomBasicTest):

    def __init__(self, env_name, in_channels, batch_size, render):
        super().__init__(env_name, in_channels, batch_size, render)


    def create_worker(self):
        raw_env = Vizdoomenvmywayhome(self.render)
        worker = SkipAndFramesEnv(raw_env, self.in_channels)

        return worker


