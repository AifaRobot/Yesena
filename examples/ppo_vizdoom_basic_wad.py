from methods import PPO
from models import ActorCriticFactory1
from workers import WorkerFactoryVizdoomBasic
from tests import VizdoomBasicTest

if __name__ == "__main__":

    arguments = {
        "env_name": "Vizdoom",
        "lr": 1e-4,
        "gamma": 0.99,
        "value_coeficient": 0.5,
        "entropy_coeficient": 0.01,
        "clip": 0.1,
        "minibatch_size": 32,
        "batch_size": 128,
        "epochs": 3,
        "episodes": 100,
        "lam": 1.0,
        "in_channels": 4,
        "n_actions": 3,
        "num_processes": 1,
    }

    method = PPO(
        main_model_factory = ActorCriticFactory1,
        worker_factory = WorkerFactoryVizdoomBasic,
        test_factory = VizdoomBasicTest,
        arguments = arguments,
    )
    
    method.train()
    method.test()
