from methods import PPO
from models import ActorCriticFactory7
from workers import WorkerFactoryVizdoomBasic
from tests import VizdoomBasicTest

if __name__ == "__main__":

    arguments = {
        "env_name": "VizdoomBasic-v0",
        "lr": 0.0001,
        "gamma": 0.99,
        "value_coeficient": 0.5,
        "entropy_coeficient": 0.01,
        "clip": 0.2,
        "minibatch_size": 64,
        "batch_size": 64,
        "epochs": 4,
        "episodes": 1000,
        "lam": 0.99,
        "in_channels": 4,
        "n_actions": 3,
        "num_processes": 4,
        "advantages_normalize": True,
        #"value_targets_normalize": True,
        #"show_processed_image": True,
        #"seed": 48
    }

    method = PPO(
        main_model_factory = ActorCriticFactory7,
        worker_factory = WorkerFactoryVizdoomBasic,
        test_factory = VizdoomBasicTest,
        arguments = arguments,
    )
    
    method.train()
    method.test()
