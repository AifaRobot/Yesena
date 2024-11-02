import numpy as np
import torch.multiprocessing as mp
from rewards import Reward, Advantage
from models import NoCuriosityFactory
from methods.utils.logger import Logger
from agents import Agent
from envs.process import process
from torch.optim import Adam
from methods.utils import SharedState

class MethodBase:
    def __init__(self, arguments, main_model_factory, worker_factory, test_factory, curiosity_model_factory = '', optimizer = '', generalized_value = '', 
        generalized_advantage = '', save_path = '', env_name = '', name = ''):

        self.env_name = env_name

        self.default_learning_rate = 0.001
        self.default_epochs = 100
        self.default_minibatch_size = 32
        self.default_batch_size = 64
        self.default_episodes = 1000
        self.default_in_channels = 3
        self.default_num_processes = 4
        self.default_clip = 0.2
        self.default_value_coeficient = 0.5
        self.default_entropy_coeficient = 0.01
        self.default_damping = 1e-3
        self.default_trust_region = 0.001
        self.default_line_search_num = 10
        self.default_k = 10
        self.default_gamma = 0.99
        self.default_lam = 1.0
        self.default_critic_loss_cliping = 'False'
        self.default_critic_loss_clip = 0.2
        self.default_verbose = 'True'
        self.default_normalize = 'False'

        self.learning_rate = getattr(arguments, 'lr', self.default_learning_rate)
        self.n_updates = getattr(arguments, 'epochs', self.default_epochs)
        self.minibatch_size = getattr(arguments, 'minibatch_size', self.default_minibatch_size)
        self.batch_size = getattr(arguments, 'batch_size', self.default_batch_size)
        self.episodes = getattr(arguments, 'episodes', self.default_episodes)
        self.in_channels = getattr(arguments, 'in_channels', self.default_in_channels)
        self.num_processes = getattr(arguments, 'num_processes', self.default_num_processes)
        self.clip = getattr(arguments, 'clip', self.default_clip)
        self.value_coeficient = getattr(arguments, 'value_coeficient', self.default_value_coeficient)
        self.entropy_coeficient = getattr(arguments, 'entropy_coeficient', self.default_entropy_coeficient)
        self.damping = getattr(arguments, 'damping', self.default_damping)
        self.trust_region = getattr(arguments, 'trust_region', self.default_trust_region)
        self.line_search_num = getattr(arguments, 'line_search_num', self.default_line_search_num)
        self.k = getattr(arguments, 'k', self.default_k)
        self.gamma = getattr(arguments, 'gamma', self.default_gamma)
        self.lam = getattr(arguments, 'lam', self.default_lam)
        self.critic_loss_cliping = getattr(arguments, 'critic_loss_cliping', self.default_critic_loss_cliping) == 'True'
        self.critic_loss_clip = getattr(arguments, 'critic_loss_clip', self.default_critic_loss_clip)
        self.verbose = getattr(arguments, 'verbose', self.default_verbose) == 'True'
        self.normalize = getattr(arguments, 'normalize', self.default_normalize) == 'True'

        self.curiosity_model_factory = NoCuriosityFactory() if(curiosity_model_factory == '') else curiosity_model_factory
        self.optimizer = Adam if(optimizer == '') else optimizer
        self.generalized_value = Reward(self.gamma) if(generalized_value == '') else generalized_value
        self.generalized_advantage = Advantage(self.gamma, self.lam, self.normalize) if(generalized_advantage == '') else generalized_advantage
        self.save_path = ('saves/' if(save_path == '') else save_path) +  name + '-' + env_name

        self.main_model_factory = main_model_factory
        self.worker_factory = worker_factory
        self.test_factory = test_factory

        self.global_agent = Agent(
            self.main_model_factory,
            self.curiosity_model_factory,
            self.save_path
        )

        self.logger = Logger(name, env_name)

    def update_metrics(self, ciclo, dict_metrics):
        self.logger.update_metrics(dict_metrics)

        if(self.verbose == True):
            print('Ciclo: ', ciclo, 'Promedio: ', np.mean(self.logger.history_extrinsic_rewards[-100:]) / self.num_processes)

    def save_models(self):
        self.global_agent.save_models()
    
    def draw_plots(self):
        self.logger.draw_plots()

    def get_agent(self):
        return self.global_agent

    def train(self):

        #mp.set_start_method('spawn')

        barrier = mp.Barrier(self.num_processes)

        lock = mp.Lock()

        shared_state = SharedState(self.num_processes, lock)

        processes = []
        for _ in range(self.num_processes):
            
            p = mp.Process(
                target=process, 
                args=(self, barrier, shared_state)
            )

            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    
    def test(self):
        self.test_factory.run(self)


