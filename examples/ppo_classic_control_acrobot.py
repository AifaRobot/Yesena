from methods import PPO
from models import ActorCriticFactory3
from workers import WorkerFactoryGymAcrobot
from tests import GymTestAcrobot

if __name__ == "__main__":

    arguments = {
        "env_name": "Acrobot-v1",
        "lr": 1e-5,
        "gamma": 0.99,
        "value_coeficient": 0.5,
        "entropy_coeficient": 0.01,
        "clip": 0.2,
        "minibatch_size": 200,
        "batch_size": 400,
        "epochs": 3,
        "episodes": 300,
        "lam": 1.0,
        "in_channels": 6,
        "n_actions": 3,
        "num_processes": 6,
    }

    method = PPO(
        main_model_factory = ActorCriticFactory3,
        worker_factory = WorkerFactoryGymAcrobot,
        test_factory = GymTestAcrobot,
        arguments = arguments,
    )
    
    method.train()
    method.test()
