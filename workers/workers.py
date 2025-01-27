import gymnasium as gym
from wrappers.skip_and_frames_env import SkipAndFramesEnv
from wrappers.gym_env import GymEnvAcrobot, GymEnvCartpole, GymEnvLunarLander, GymEnvCarRacing, GymEnvVizdoomBasic, GymEnvVizdoomMyWayHome
from workers.worker import Worker
from vizdoom import gymnasium_wrapper

'''
    Here are all those in charge of creating the Workers. Each Worker needs a different configuration depending
'''

class WorkerFactory:
    def __init__(self, env_name, in_channels, batch_size, optional_params):
        self.env_name = env_name
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.optional_params = optional_params

    @property
    def create(self):
        raise NotImplementedError('Implement me')

class WorkerFactoryVizdoomBasic(WorkerFactory):

    def __init__(self, env_name, in_channels, batch_size, optional_params = {}):
        super().__init__(env_name, in_channels, batch_size, optional_params)

    def create(self, agent):
        raw_env = gym.make(self.env_name)
        env = GymEnvVizdoomBasic(raw_env, self.optional_params)
        env = SkipAndFramesEnv(env, self.in_channels, (32, 32), 60, 140, optional_params = self.optional_params)
        worker = Worker(env, agent, self.batch_size)

        return worker

class WorkerFactoryVizdoomMyWayHome(WorkerFactory):

    def __init__(self, env_name, in_channels, batch_size, optional_params = {}):
        super().__init__(env_name, in_channels, batch_size, optional_params)

    def create(self, agent):
        raw_env = gym.make(self.env_name)
        env = GymEnvVizdoomMyWayHome(raw_env, self.optional_params)
        env = SkipAndFramesEnv(env, self.in_channels, (42, 42), optional_params = self.optional_params)
        worker = Worker(env, agent, self.batch_size)

        return worker

class WorkerFactoryGymAcrobot(WorkerFactory):

    def __init__(self, env_name, in_channels, batch_size, optional_params = {}):
        super().__init__(env_name, in_channels, batch_size, optional_params)

    def create(self, agent):
        raw_env = gym.make(self.env_name)
        env = GymEnvAcrobot(raw_env, self.optional_params)

        worker = Worker(env, agent, self.batch_size)

        return worker

class WorkerFactoryGymCartpole(WorkerFactory):

    def __init__(self, env_name, in_channels, batch_size, optional_params = {}):
        super().__init__(env_name, in_channels, batch_size, optional_params)

    def create(self, agent):
        raw_env = gym.make(self.env_name)
        env = GymEnvCartpole(raw_env, self.optional_params)

        worker = Worker(env, agent, self.batch_size)

        return worker

class WorkerFactoryGymLunarLander(WorkerFactory):

    def __init__(self, env_name, in_channels, batch_size, optional_params = {}):
        super().__init__(env_name, in_channels, batch_size, optional_params)

    def create(self, agent):
        raw_env = gym.make(self.env_name)
        env = GymEnvLunarLander(raw_env, self.optional_params)

        worker = Worker(env, agent, self.batch_size)

        return worker

class WorkerFactoryGymCarRacing(WorkerFactory):

    def __init__(self, env_name, in_channels, batch_size, optional_params = {}):
        super().__init__(env_name, in_channels, batch_size, optional_params)
        self.continuous = True

    def create(self, agent):
        raw_env = gym.make(self.env_name)
        env = GymEnvCarRacing(raw_env, self.optional_params)
        env = SkipAndFramesEnv(env, self.in_channels, (42, 42), None, 80)

        worker = Worker(env, agent, self.batch_size)

        return worker
