import argparse
from methods import PPO
from models import ActorCriticFactory3
from workers import WorkerFactoryGymCartpole
from tests import GymTestCartpole
from rewards import Advantage

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()

    env_name = 'CartPole-v1'

    args.add_argument('-lr', type=float, default=1e-4)
    args.add_argument('-gamma', type=float, default=0.99)
    args.add_argument('-c1', type=float, default=1.0)
    args.add_argument('-c2', type=float, default=0.01)
    args.add_argument('-clip', type=float, default=0.2)
    args.add_argument('-minibatch_size', type=int, default=128)
    args.add_argument('-batch_size', type=int, default=128)
    args.add_argument('-epochs', type=int, default=3)
    args.add_argument('-train', choices=('True','False'), default='True')
    args.add_argument('-episodes', type=int, default=600)
    args.add_argument('-lam', type=float, default=0.9)
    args.add_argument('-in_channels', type=int, default=4)
    args.add_argument('-n_actions', type=int, default=2)
    args.add_argument('-num_processes', type=int, default=6)
    args.add_argument('-alpha', type=float, default=0.1)
    args.add_argument('-beta', type=float, default=0.2)
  
    arguments = args.parse_args()

    method = PPO(
        env_name = env_name,
        main_model_factory = ActorCriticFactory3(arguments.in_channels, arguments.n_actions),
        worker_factory = WorkerFactoryGymCartpole(env_name, arguments.in_channels, arguments.batch_size),
        test_factory = GymTestCartpole(env_name, arguments.in_channels, arguments.batch_size, True),
        generalized_advantage = Advantage(arguments.gamma, arguments.lam, True),
        arguments = arguments,
    )
    
    if(arguments.train == 'True'):
        print('Modo: Entrenamiento')
        method.train()
    else:
        print('Modo: Testeo')
        method.test()