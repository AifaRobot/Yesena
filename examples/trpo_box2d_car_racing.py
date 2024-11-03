from methods import TRPO
from models import ActorCriticFactory6
from workers import WorkerFactoryGymCarRacing
from tests import GymTestCarRacing

if __name__ == "__main__":

    arguments = {
        "env_name": "CarRacing-v3",
        "lr": 1e-4,
        "gamma": 0.99,
        "value_coeficient": 0.5,
        "entropy_coeficient": 0.01,
        "clip": 0.1,
        "minibatch_size": 128,
        "batch_size": 128,
        "epochs": 10,
        "episodes": 1500,
        "lam": 1.0,
        "in_channels": 4,
        "n_actions": 5,
        "num_processes": 1,
        "damping": 0.1,
        "trust_region": 0.001,
        "line_search_num": 10,
        "k": 10,
    }

    method = TRPO(
        main_model_factory = ActorCriticFactory6,
        worker_factory = WorkerFactoryGymCarRacing,
        test_factory = GymTestCarRacing,
        arguments = arguments,
    )

    method.train()
    method.test()
