import gymnasium as gym
import torch
import cv2
import numpy as np

'''
    Here you will find the wrappers for all the environments that Yesena supports. Each wrapper functions 
    as a connection between the game and the agent, allowing the responses given by the environment to become 
    something more understandable to the agent.
'''

class GymEnv(gym.Wrapper):
    def __init__(self, env, optional_params = {}):
        super(GymEnv, self).__init__(env)
        self.number_step = 0
        self.seed = optional_params.get('seed', '')
    
    def reset(self, **kwargs):
        if(self.seed != ''):
            kwargs['seed'] = self.seed

        observation, _ = self.env.reset(**kwargs)
        self.number_step = 0

        return torch.tensor(observation)

class GymEnvCartpole(GymEnv):
    def __init__(self, env, optional_params = {}):
        super(GymEnvCartpole, self).__init__(env, optional_params)

    def step(self, action):
        observation, reward, done, _, _ = self.env.step(action)
        self.number_step = self.number_step + 1

        if done:
            reward = -1
            self.number_step = 0
        else:
            reward = 0.1

        if (self.number_step == 3000):
            done = True
            self.number_step = 0

        return torch.from_numpy(observation).float(), reward, done

class GymEnvAcrobot(GymEnv):
    def __init__(self, env, optional_params = {}):
        super(GymEnvAcrobot, self).__init__(env, optional_params)

    def step(self, action):
        observation, reward, done, _, _ = self.env.step(action)

        return torch.tensor(observation), reward, done

class GymEnvLunarLander(GymEnv):
    def __init__(self, env, optional_params = {}):
        super(GymEnvLunarLander, self).__init__(env, optional_params)

    def step(self, action):
        observation, reward, done, _, _ = self.env.step(action)

        return torch.tensor(observation), reward, done

class GymEnvCarRacing(GymEnv):
    def __init__(self, env, optional_params = {}):
        super(GymEnvCarRacing, self).__init__(env, optional_params)
        self.env = env
        self.done_before = False

        self.tiles = 0
        self.steps = 0
        self.actions = [[0.0, 0.01, 0.0], [0.2, 0.01, 0.0], [-0.2, 0.01, 0.0], [0.4, 0.01, 0.0], [-0.4, 0.01, 0.0]]

    def reset(self, **kwargs):
        if(self.seed != ''):
            kwargs['seed'] = self.seed

        observation, _ = self.env.reset(**kwargs)
        self.steps = 0
        self.tiles = 0

        return observation

    def step(self, action):
        observation, reward, done, info, _ = self.env.step(np.array(self.actions[action]))
        
        self.steps += 1
        
        if(self.steps > 10 and self.is_off_track(observation)):
            done = True
            
        if(reward > 0):
            self.tiles += 1

        reward = -1 if done else 0.8 + 0.2 * (self.tiles / 732) 

        return observation, reward, done, info

        
    def is_off_track(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        car_x, car_y = 48, 68
        car_area = binary[car_y-5:car_y+5, car_x-5:car_x+5]
        
        non_track_pixels = np.sum(car_area == 255)

        if non_track_pixels > 70:
            return True
        
        return False

class GymEnvVizdoomBasic(GymEnv):
    def __init__(self, env, optional_params = {}):
        super(GymEnvVizdoomBasic, self).__init__(env, optional_params)
        self.actions = [3, 1, 2]

    def reset(self, **kwargs):
        if(self.seed != ''):
            kwargs['seed'] = self.seed
        
        observation, _ = self.env.reset(**kwargs)

        return observation['screen']

    def step(self, action):

        action = self.actions[action]

        observation, reward, done, info, _ = self.env.step(action)

        return observation['screen'], reward, done, info

class GymEnvVizdoomMyWayHome(GymEnv):
    def __init__(self, env, optional_params = {}):
        super(GymEnvVizdoomMyWayHome, self).__init__(env, optional_params)
        self.actions = [3, 4, 5]

    def reset(self, **kwargs):
        if(self.seed != ''):
            kwargs['seed'] = self.seed

        observation, _ = self.env.reset(**kwargs)

        return observation['screen']

    def step(self, action):
        action = self.actions[action]

        observation, reward, done, info, _ = self.env.step(action)

        return observation['screen'], reward, done, info
