import gymnasium as gym
from agents import Agent
from wrappers.gym_env import GymEnvCartpole, GymEnvAcrobot, GymEnvLunarLander, GymEnvCarRacing, GymEnvVizdoomBasic, GymEnvVizdoomMyWayHome
from wrappers.skip_and_frames_env import SkipAndFramesEnv

'''
    The agents' evidence is saved in this file. When an agent finishes training in an environment, a test can be run to 
    demonstrate how the agent performs.

    Each test has the run method that is executed to start the loop where the agent will play the environment.

    The fundamental difference between the testing period and the training period is that, in training, the agent in each 
    state chooses an action using the probabilities provided by the agent, but having a certain randomness. However, in the 
    trial period the action that will be chosen in each state is always the one with the highest probability. This is because 
    training must have a certain randomness when choosing an action because the agent must explore all possible actions to know 
    the path that gives the greatest reward. On the other hand, in the testing period there is no need for randomness because 
    we have already trained and therefore the actions should only be those that the agent is mostly sure will be beneficial.

    Let's look at an example:

    We have an agent who is in a certain state and has to choose between two actions. After processing that state, it returns 
    the following probabilities: [0.75, 0.25]:

    * In the training period the agent will choose one of the two actions at random, but taking these probabilities into 
    account, that is, there will be a 75% probability that the first action will be chosen and a 25% probability that the 
    second will be chosen. .

    * In the trial period, the action with the highest probability will be chosen because it is the one that is most likely 
    to return the highest reward throughout the episode. That is to say, it will choose the first because, since it is the 
    one with the greatest probability, it must be the one that returns a greater reward in the long term.
'''

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


class GymTestLunarLander():

    def __init__(self, env_name, in_channels, batch_size, render):
        self.env_name = env_name
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.render = render

        raw_env = gym.make(self.env_name, render_mode="human")
        self.worker = GymEnvLunarLander(raw_env)

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


class GymTestCarRacing():

    def __init__(self, env_name, in_channels, batch_size, render):
        self.env_name = env_name
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.render = render

        raw_env = gym.make(self.env_name, render_mode="human")
        env = GymEnvCarRacing(raw_env)
        self.worker = SkipAndFramesEnv(env, self.in_channels, (42, 42), None, 80)

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


class VizdoomBasicTest():

    def __init__(self, env_name, in_channels, batch_size, render):
        self.env_name = env_name
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.render = render

    def run(self, method):
        optional_params = { "show_processed_image": True }

        raw_env = gym.make(self.env_name, render_mode="human")
        env = GymEnvVizdoomBasic(raw_env, optional_params)
        worker = SkipAndFramesEnv(env, self.in_channels, (32, 32), 60, 140, optional_params)

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
                action, next_hx = agente.get_action_max_prob(observation, hx)

                next_state, reward, done = worker.step(action.item())

                observation = next_state
                hx = next_hx

                if done:
                    print('Gano' if reward > 0 else 'Perdio')


class VizdoomMyWayHomeTest():

    def __init__(self, env_name, in_channels, batch_size, render):
        self.env_name = env_name
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.render = render

    def run(self, method):
        optional_params = { "show_processed_image": True, "seed": 18 }

        raw_env = gym.make(self.env_name, render_mode="human")
        env = GymEnvVizdoomMyWayHome(raw_env, optional_params)
        worker = SkipAndFramesEnv(env, self.in_channels, (42, 42), optional_params = optional_params)

        agente = Agent(
            method.main_model_factory,
            method.curiosity_model_factory,
            method.save_path
        )
        
        while True:
            done = False
            observation = worker.reset()
            hx = agente.get_new_hx()

            max_reward = 0

            while not done: 
                action, next_hx = agente.get_action_max_prob(observation, hx)

                next_state, reward, done = worker.step(action.item())

                observation = next_state
                hx = next_hx

                if(reward > max_reward):
                    max_reward = reward

                if done:
                    print('Gano' if max_reward > 0.8 else 'Perdio')