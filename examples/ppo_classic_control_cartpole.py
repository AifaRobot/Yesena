from methods import PPO
from models import ActorCriticFactory3
from workers import WorkerFactoryGymCartpole
from tests import GymTestCartpole

if __name__ == "__main__":

    arguments = {
        "env_name": "CartPole-v1",
        "lr": 1e-4,
        "gamma": 0.999,
        "value_coeficient": 0.5,
        "entropy_coeficient": 0.01,
        "clip": 0.2,
        "minibatch_size": 128,
        "batch_size": 128,
        "epochs": 10,
        "episodes": 600,
        "lam": 1.0,
        "in_channels": 4,
        "n_actions": 2,
        "num_processes": 4,
    }

    method = PPO(
        main_model_factory = ActorCriticFactory3,
        worker_factory = WorkerFactoryGymCartpole,
        test_factory = GymTestCartpole,
        arguments = arguments,
    )

    method.train()
    method.test()
