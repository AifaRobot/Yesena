from methods import TRPO
from models import ActorCriticFactory5
from workers import WorkerFactoryGymLunarLander
from tests import GymTestLunarLander

if __name__ == "__main__":

    arguments = {
        "env_name": "LunarLander-v3",
        "lr": 1e-4,
        "gamma": 0.999,
        "value_coeficient": 0.5,
        "entropy_coeficient": 0.01,
        "clip": 0.1,
        "minibatch_size": 128,
        "batch_size": 1024,
        "epochs": 4,
        "episodes": 5000,
        "lam": 1.0,
        "in_channels": 8,
        "n_actions": 4,
        "num_processes": 2,
        "damping": 0.1,
        "trust_region": 0.001,
        "line_search_num": 10,
        "k": 10,
    }

    method = TRPO(
        main_model_factory = ActorCriticFactory5,
        worker_factory = WorkerFactoryGymLunarLander,
        test_factory = GymTestLunarLander,
        arguments = arguments,
    )

    method.train()
    method.test()
