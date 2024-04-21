import gym
import gym_miniworld
from wrappers.skip_and_frames_env import SkipAndFramesEnv
from wrappers.vizdoomenv_basic import Vizdoomenv
from wrappers.vizdoomenv_my_way_home import Vizdoomenvmywayhome
from wrappers.gym_env import GymEnvAcrobot, GymEnvCartpole
from workers.worker import Worker

class WorkerFactory:
    def __init__(self, env_name, in_channels, batch_size):
        self.env_name = env_name
        self.in_channels = in_channels
        self.batch_size = batch_size

    @property
    def create(self):
        raise NotImplementedError('Implement me')

class WorkerFactoryMiniworld(WorkerFactory):

    def __init__(self, env_name, in_channels, batch_size):
        super().__init__(env_name, in_channels, batch_size)

    def create(self, agent):
        raw_env = gym.make(self.env_name)
        env = SkipAndFramesEnv(raw_env, self.in_channels)
        worker = Worker(env, agent, self.batch_size)

        return worker

class WorkerFactoryVizdoomBasic(WorkerFactory):

    def __init__(self, env_name, in_channels, batch_size, render):
        super().__init__(env_name, in_channels, batch_size)
        self.render = render

    def create(self, agent):
        raw_env = Vizdoomenv(self.render)
        env = SkipAndFramesEnv(raw_env, self.in_channels)
        worker = Worker(env, agent, self.batch_size)

        return worker

class WorkerFactoryVizdoomMyWayHome(WorkerFactory):

    def __init__(self, env_name, in_channels, batch_size, render):
        super().__init__(env_name, in_channels, batch_size)
        self.render = render

    def create(self, agent):
        raw_env = Vizdoomenvmywayhome(self.render)
        env = SkipAndFramesEnv(raw_env, self.in_channels)
        worker = Worker(env, agent, self.batch_size)

        return worker

class WorkerFactoryGymAcrobot(WorkerFactory):

    def __init__(self, env_name, in_channels, batch_size):
        super().__init__(env_name, in_channels, batch_size)

    def create(self, agent):
        raw_env = gym.make(self.env_name)
        env = GymEnvAcrobot(raw_env)

        worker = Worker(env, agent, self.batch_size)

        return worker

class WorkerFactoryGymCartpole(WorkerFactory):

    def __init__(self, env_name, in_channels, batch_size):
        super().__init__(env_name, in_channels, batch_size)

    def create(self, agent):
        raw_env = gym.make(self.env_name)
        env = GymEnvCartpole(raw_env)

        worker = Worker(env, agent, self.batch_size)

        return worker