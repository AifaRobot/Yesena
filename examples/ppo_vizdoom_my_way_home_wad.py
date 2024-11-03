from methods import PPO
from models import ActorCriticFactory1, CuriosityFactory
from workers import WorkerFactoryVizdoomMyWayHome
from tests import VizdoomMyWayHomeTest

if __name__ == "__main__":

    arguments = {
        "env_name": "Vizdoom_my_way_home",
        "lr": 1e-4,
        "gamma": 0.99,
        "value_coeficient": 0.5,
        "entropy_coeficient": 0.01,
        "clip": 0.1,
        "minibatch_size": 128,
        "batch_size": 256,
        "epochs": 3,
        "episodes": 2000,
        "lam": 1.0,
        "in_channels": 4,
        "n_actions": 3,
        "num_processes": 6,
        "alpha": 0.1,
        "beta": 0.2,
    }

    method = PPO(
        main_model_factory = ActorCriticFactory1,
        curiosity_model_factory = CuriosityFactory,
        worker_factory = WorkerFactoryVizdoomMyWayHome,
        test_factory = VizdoomMyWayHomeTest,
        arguments = arguments,
    )

    method.train()
    method.test()
