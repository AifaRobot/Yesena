import gymnasium as gym
#import gym_miniworld
import time
from wrappers.skip_and_frames_env import SkipAndFramesEnv
from agents import Agent


class MiniworldTest():

    def __init__(self, env_name, in_channels, batch_size, render):
        self.env_name = env_name
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.render = render

    def run(self, method):   
        raw_env = gym.make(self.env_name)
        worker = SkipAndFramesEnv(raw_env, self.in_channels)

        agente = Agent(
            method.main_model_factory,
            method.curiosity_model_factory,
            method.save_path
        )
        
        while True:
            done = False
            observation = worker.reset()

            hx = agente.get_new_hx()

            while not done: 
                worker.render()
                time.sleep(0.1)
                
                action, next_hx = agente.get_action_max_prob(observation, hx)

                next_state, reward, done = worker.step(action.item())

                observation = next_state
                hx = next_hx

                if done:
                    print('Gano' if reward > 0 else 'Perdio')

