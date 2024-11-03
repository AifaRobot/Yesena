from methods.ppo import PPO
from models import ActorCriticFactory3
from workers import WorkerFactoryGymLunarLander
from tests import GymTestLunarLander

if __name__ == "__main__":
    
    arguments = {
        "env_name": "LunarLander-v3",
        "lr": 0.0001,
        "gamma": 0.99,
        "value_coeficient": 0.5,
        "entropy_coeficient": 0.01,
        "clip": 0.1,
        "minibatch_size": 32,
        "batch_size": 128,
        "epochs": 4,
        "episodes": 2000,
        "lam": 1.0,
        "in_channels": 8,
        "n_actions": 4,
        "num_processes": 5,
    }

    method = PPO(
        main_model_factory = ActorCriticFactory3,
        worker_factory = WorkerFactoryGymLunarLander,
        test_factory = GymTestLunarLander,
        arguments = arguments,
    )

    method.train()
    method.test()
