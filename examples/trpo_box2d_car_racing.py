import argparse
from methods import TRPO
from models import ActorCriticFactory6
from workers import WorkerFactoryGymCarRacing
from tests import GymTestCarRacing

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()

    env_name = 'CarRacing-v3'

    args.add_argument('-lr', type=float, default=0.0001)
    args.add_argument('-gamma', type=float, default=0.99)
    args.add_argument('-value_coeficient', type=float, default=0.5)
    args.add_argument('-entropy_coeficient', type=float, default=0.01)
    args.add_argument('-clip', type=float, default=0.1)
    args.add_argument('-minibatch_size', type=int, default=128)
    args.add_argument('-batch_size', type=int, default=128)
    args.add_argument('-epochs', type=int, default=10)
    args.add_argument('-train', choices=('True','False'), default='True')
    args.add_argument('-episodes', type=int, default=1500)
    args.add_argument('-lam', type=float, default=1.0)
    args.add_argument('-in_channels', type=int, default=4)
    args.add_argument('-n_actions', type=int, default=5)
    args.add_argument('-num_processes', type=int, default=1)
    args.add_argument('-damping', type=float, default=0.1)
    args.add_argument('-trust_region', type=float, default=0.001)
    args.add_argument('-line_search_num', type=int, default=10)
    args.add_argument('-k', type=int, default=10)

    arguments = args.parse_args()

    method = TRPO(
        env_name = env_name,
        main_model_factory = ActorCriticFactory6(arguments.in_channels, arguments.n_actions),
        worker_factory = WorkerFactoryGymCarRacing(env_name, arguments.in_channels, arguments.batch_size, True),
        test_factory = GymTestCarRacing(env_name, arguments.in_channels, arguments.batch_size, True),
        arguments = arguments,
    )
    
    if(arguments.train == 'True'):
        print('Modo: Entrenamiento')
        method.train()
    else:
        print('Modo: Testeo')
        method.test()