import gym
import cv2
import torch
import collections

class SkipAndFramesEnv(gym.Wrapper):
    def __init__(self, env, num_frames = 4):
        super(SkipAndFramesEnv, self).__init__(env)

        self.num_frames = num_frames
        self.frames_stack = collections.deque(maxlen=num_frames)
    
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = self.process_img(observation)

        self.frames_stack.clear()

        for _ in range(self.num_frames):
            self.frames_stack.append(observation)

        observation = torch.tensor(self.frames_stack, dtype=torch.float32).unsqueeze(0)

        return observation

    def step(self, action):
        total_reward = 0

        for _ in range(self.num_frames):
            observation, reward, done, _ = self.env.step(action)
            total_reward += reward

            if done:
                break
        
        observation = self.process_img(observation)

        self.frames_stack.append(observation)

        #if total_reward > 0:
        #    print('La recompensa fue mayor a cero y fue: ', total_reward)

        observation = torch.tensor(self.frames_stack, dtype=torch.float32).unsqueeze(0)

        return observation, total_reward, done

    def process_img(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (42,42), interpolation=cv2.INTER_AREA)
        image = image / 255

        return  image

