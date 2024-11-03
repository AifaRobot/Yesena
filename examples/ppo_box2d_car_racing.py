from methods import PPO
from models import ActorCriticFactory4
from workers import WorkerFactoryGymCarRacing
from tests import GymTestCarRacing

if __name__ == "__main__":
    
    arguments = {
        "env_name": "CarRacing-v3",
        "lr": 0.0001,
        "gamma": 0.99,
        "value_coeficient": 0.5,
        "entropy_coeficient": 0.01,
        "clip": 0.1,
        "minibatch_size": 128,
        "batch_size": 128,
        "epochs": 10,
        "episodes": 2500,
        "lam": 1.0,
        "in_channels": 4,
        "n_actions": 5,
        "num_processes": 2,
    }

    method = PPO(
        main_model_factory = ActorCriticFactory4,
        worker_factory = WorkerFactoryGymCarRacing,
        test_factory = GymTestCarRacing,
        arguments = arguments,
    )

    method.train()
    method.test()
