import gym
import torch

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
        super().__init__(env)

    def step(self, action):
        observation, reward, done, _, _ = self.env.step(action)
        self.number_step = self.number_step + 1

        if done:
            reward = -1
        else:
            reward = 0

        if(self.number_step == 700):
            done = True
            self.number_step = 0
            reward = 1

        return torch.tensor(observation), reward, done

class GymEnvAcrobot(GymEnv):
    def __init__(self, env):
        super(GymEnvAcrobot, self).__init__(env)

    def step(self, action):
        observation, reward, done, _, _ = self.env.step(action)
        self.number_step = self.number_step + 1

        if(reward == -1):
            reward = 0
        else:
            reward = 1

        if (self.number_step > 350):
            done = True
            self.number_step  = 0
            reward = -1

        return torch.tensor(observation), reward, done
