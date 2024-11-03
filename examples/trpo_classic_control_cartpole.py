from methods import TRPO
from models import ActorCriticFactory5
from workers import WorkerFactoryGymCartpole
from tests import GymTestCartpole

if __name__ == "__main__":

    arguments = {
        "env_name": "CartPole-v1",
        "lr": 1e-5,
        "gamma": 0.999,
        "value_coeficient": 0.5,
        "entropy_coeficient": 0.01,
        "clip": 0.1,
        "minibatch_size": 128,
        "batch_size": 128,
        "epochs": 10,
        "episodes": 2000,
        "lam": 1.0,
        "in_channels": 4,
        "n_actions": 2,
        "num_processes": 3,
        "damping": 1e-3,
        "trust_region": 0.001,
        "line_search_num": 10,
        "k": 10,
        "normalize": True,
    }

    method = TRPO(
        main_model_factory = ActorCriticFactory5,
        worker_factory = WorkerFactoryGymCartpole,
        test_factory = GymTestCartpole,
        arguments = arguments,
    )

    method.train()
    method.test()
