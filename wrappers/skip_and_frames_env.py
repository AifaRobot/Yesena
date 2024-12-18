import gymnasium as gym
import cv2
import torch
import collections
import numpy as np

class SkipAndFramesEnv(gym.Wrapper):
    def __init__(self, env, num_frames = 4, resize_image = (42, 42), initial_height = None, final_height = None, optional_params = {}):
        super(SkipAndFramesEnv, self).__init__(env)

        self.num_frames = num_frames
        self.resize_image = resize_image
        self.frames_stack = collections.deque(maxlen=num_frames)
        self.initial_height = initial_height
        self.final_height = final_height
        self.show_processed_image = optional_params.get('show_processed_image', False)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = self.process_img(observation)

        self.frames_stack.clear()

        for _ in range(self.num_frames):
            self.frames_stack.append(observation)

        observation = torch.tensor(np.array(self.frames_stack), dtype=torch.float32).unsqueeze(0)

        return observation

    def step(self, action):
        total_reward = 0

        for _ in range(self.num_frames):
            observation, reward, done, _ = self.env.step(action)

            total_reward += reward

            if done:
                break

        observation = self.process_img(observation)

        if(self.show_processed_image):
            self.show_img(observation)

        self.frames_stack.append(observation)

        observation = torch.tensor(np.array(self.frames_stack), dtype=torch.float32).unsqueeze(0)

        return observation, total_reward, done

    def process_img(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if(self.initial_height != None):
            image = image[self.initial_height:]

        if(self.final_height != None):
            image = image[:self.final_height]

        image = cv2.resize(image, self.resize_image, interpolation=cv2.INTER_AREA)
        image = image / 255

        return  image

    def show_img(self, observation):
        window_title = "Juego"

        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)        
        cv2.imshow(window_title, observation)
        
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
