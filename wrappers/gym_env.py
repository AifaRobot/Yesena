import gymnasium as gym
import torch
import cv2
import numpy as np

'''
    Aquí encontrarás los wrappers para todos los entornos que soporta Yesena.
    Cada wrapper funciona como una conexión entre el juego y el agente, lo que permite que las respuestas dadas por 
    elentorno se conviertan en algo mas comprensible para el agente.
'''

class GymEnv(gym.Wrapper):
    def __init__(self, env):
        super(GymEnv, self).__init__(env)
        self.number_step = 0
    
    def reset(self, **kwargs):
        observation, _ = self.env.reset(**kwargs)
        self.number_step = 0

        return torch.tensor(observation)

class GymEnvCartpole(GymEnv):
    def __init__(self, env):
        super(GymEnvCartpole, self).__init__(env)

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
    def __init__(self, env):
        super(GymEnvAcrobot, self).__init__(env)

    def step(self, action):
        observation, reward, done, _, _ = self.env.step(action)

        return torch.tensor(observation), reward, done

class GymEnvLunarLander(GymEnv):
    def __init__(self, env):
        super(GymEnvLunarLander, self).__init__(env)

    def step(self, action):
        observation, reward, done, _, _ = self.env.step(action)

        return torch.tensor(observation), reward, done

class GymEnvCarRacing(GymEnv):
    def __init__(self, env):
        super(GymEnvCarRacing, self).__init__(env)
        self.env = env
        self.done_before = False

        self.tiles = 0
        self.steps = 0
        self.actions = [[0.0, 0.01, 0.0], [0.2, 0.01, 0.0], [-0.2, 0.01, 0.0], [0.4, 0.01, 0.0], [-0.4, 0.01, 0.0]]

    def reset(self, **kwargs):
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

class GymEnvMiniworld(GymEnv):
    def __init__(self, env):
        super(GymEnvMiniworld, self).__init__(env)

    def reset(self, **kwargs):
        observation, _ = self.env.reset(**kwargs)

        return observation

    def step(self, action):
        observation, reward, done, info, _ = self.env.step(action)

        return observation, reward, done, info