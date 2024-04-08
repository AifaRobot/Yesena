import argparse
from methods import PPO
from optimizers import SharedAdam
from rewards import Reward, Advantage
from models import ActorCriticFactory, NoCuriosityFactory
from workers import WorkerFactoryVizdoomBasic
from tests import VizdoomBasicTest

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()

    env_name = 'Vizdoom'

    args.add_argument('-lr', type=float, default=1e-4)
    args.add_argument('-gamma', type=float, default=0.99)
    args.add_argument('-c1', type=float, default=0.5)
    args.add_argument('-c2', type=float, default=0.01)
    args.add_argument('-clip', type=float, default=0.1)
    args.add_argument('-minibatch_size', type=int, default=32)
    args.add_argument('-batch_size', type=int, default=128)
    args.add_argument('-epochs', type=int, default=3)
    args.add_argument('-train', choices=('True','False'), default='True')
    args.add_argument('-episodes', type=int, default=100)
    args.add_argument('-lam', type=float, default=1.0)
    args.add_argument('-in_channels', type=int, default=4)
    args.add_argument('-n_actions', type=int, default=3)
    args.add_argument('-num_processes', type=int, default=1)
    args.add_argument('-alpha', type=float, default=0.1)
    args.add_argument('-beta', type=float, default=0.2)
  
    arguments = args.parse_args()

    method = PPO(
        env_name = env_name,
        main_model_factory = ActorCriticFactory(arguments.in_channels, arguments.n_actions),
        curiosity_model_factory = NoCuriosityFactory(),
        arguments = arguments,
        optimizer = SharedAdam,
        generalized_value = Reward(arguments.gamma),
        generalized_advantage = Advantage(arguments.gamma, arguments.lam),
        worker_factory = WorkerFactoryVizdoomBasic(env_name, arguments.in_channels, arguments.batch_size, False),
        test_factory = VizdoomBasicTest(env_name, arguments.in_channels, arguments.batch_size, True),
        save_path = 'saves/'
    )

    if(arguments.train == 'True'):
        print('Modo: Entrenamiento')
        method.train()
    else:
        print('Modo: Testeo')
        method.test()