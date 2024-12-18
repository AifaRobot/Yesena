import gymnasium as gym
from agents import Agent
from wrappers.gym_env import GymEnvCartpole, GymEnvAcrobot, GymEnvLunarLander, GymEnvCarRacing, GymEnvVizdoomBasic, GymEnvVizdoomMyWayHome
from wrappers.skip_and_frames_env import SkipAndFramesEnv

'''
    En este archivo se guardan las pruebas de los agentes. Cuando un agente termina de entrenar en un entorno, se podrá ejecutar  
    un test que permitirá demostrar cómo se desempeña el agente. 

    Cada test tiene el método run que se ejecuta para iniciar el bucle donde el agente jugara el entorno. 

    La diferencia fundamental entre el periodo de prueba y el de entrenamiento es que, en el entrenamiento, el agente en cada estado elige una 
    acción usando las probabilidades que le brinda el agente, pero teniendo cierta aleatoriedad. Sin embargo, en el periodo de prueba la acción que se 
    elegirá en cada estado siempre es la que tiene mayor probabilidad. Esto se debe a que en el entrenamiento debe tener cierta aleatoriedad al  
    momento de elegir una acción porque el agente debe explorar todas las posibles acciones para conocer el camino que le de mayor recompensa. 
    En cambio, en el periodo de prueba no es necesario que haiga aleatoriedad porque ya hemos entrenado y por lo tanto las acciones deben ser 
    solamente aquellas que la agente esta mayormente seguro que serán beneficiosas. 

    Veamos un ejemplo: 

    Tenemos un agente que se encuentra en un determinado estado y tiene que elegir entre dos acciones. Luego de procesar ese estado, devuelve  
    las siguientes probabilidades: [0.75, 0.25]: 

    * En el periodo entrenamiento el agente elegirá una de las dos acciones de manera aleatoria, pero teniendo en cuenta esas probabilidades, 
    es decir que habrá una probabilidad del 75% de que se elija la primera acción y un 25% de que se elija la segunda.  

    * En el periodo de prueba se elegirá aquella acción con mayor probabilidad porque es aquella que es más seguro que devuelva la recompensa más alta a  
    lo largo del episodio. Es decir que elegirá la primera porque al ser la que tiene mayor probabilidad, debe ser la que devuelva una  
    recompensa mayor a largo plazo. 
'''

class GymTestAcrobot():

    def __init__(self, env_name, in_channels, batch_size, render):
        self.env_name = env_name
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.render = render

        raw_env = gym.make(self.env_name, render_mode="human")
        self.worker = GymEnvAcrobot(raw_env)

    def run(self, method):   
        agente = Agent(
            method.main_model_factory,
            method.curiosity_model_factory,
            method.save_path
        )

        while True:
            done = False
            observation = self.worker.reset()

            hx = agente.get_new_hx()
            
            while not done: 
                action, next_hx = agente.get_action_max_prob(observation, hx)

                next_state, _, done = self.worker.step(action.item())

                observation = next_state
                hx = next_hx


class GymTestCartpole():

    def __init__(self, env_name, in_channels, batch_size, render):
        self.env_name = env_name
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.render = render

        raw_env = gym.make(self.env_name, render_mode="human")
        self.worker = GymEnvCartpole(raw_env)

    def run(self, method):   
        
        agente = Agent(
            method.main_model_factory,
            method.curiosity_model_factory,
            method.save_path
        )

        while True:
            done = False
            observation = self.worker.reset()

            hx = agente.get_new_hx()
            while not done: 
                action, next_hx = agente.get_action_max_prob(observation, hx)

                next_state, _, done = self.worker.step(action.item())

                observation = next_state
                hx = next_hx


class GymTestLunarLander():

    def __init__(self, env_name, in_channels, batch_size, render):
        self.env_name = env_name
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.render = render

        raw_env = gym.make(self.env_name, render_mode="human")
        self.worker = GymEnvLunarLander(raw_env)

    def run(self, method):   
        
        agente = Agent(
            method.main_model_factory,
            method.curiosity_model_factory,
            method.save_path
        )

        while True:
            done = False
            observation = self.worker.reset()

            hx = agente.get_new_hx()
            while not done: 
                action, next_hx = agente.get_action_max_prob(observation, hx)

                next_state, _, done = self.worker.step(action.item())

                observation = next_state
                hx = next_hx


class GymTestCarRacing():

    def __init__(self, env_name, in_channels, batch_size, render):
        self.env_name = env_name
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.render = render

        raw_env = gym.make(self.env_name, render_mode="human")
        env = GymEnvCarRacing(raw_env)
        self.worker = SkipAndFramesEnv(env, self.in_channels, (42, 42), None, 80)

    def run(self, method):   

        agente = Agent(
            method.main_model_factory,
            method.curiosity_model_factory,
            method.save_path
        )

        while True:
            done = False
            observation = self.worker.reset()

            hx = agente.get_new_hx()
            while not done: 
                action, next_hx = agente.get_action_max_prob(observation, hx)

                next_state, _, done = self.worker.step(action.item())

                observation = next_state
                hx = next_hx


class VizdoomBasicTest():

    def __init__(self, env_name, in_channels, batch_size, render):
        self.env_name = env_name
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.render = render

    def run(self, method):
        optional_params = { "show_processed_image": True }

        raw_env = gym.make(self.env_name, render_mode="human")
        env = GymEnvVizdoomBasic(raw_env, optional_params)
        worker = SkipAndFramesEnv(env, self.in_channels, (32, 32), 60, 140, optional_params)

        agente = Agent(
            method.main_model_factory,
            method.curiosity_model_factory,
            method.save_path
        )
        
        while True:
            done = False
            observation = worker.reset()
            hx = agente.get_new_hx()

            while not done: 
                action, next_hx = agente.get_action_max_prob(observation, hx)

                next_state, reward, done = worker.step(action.item())

                observation = next_state
                hx = next_hx

                if done:
                    print('Gano' if reward > 0 else 'Perdio')


class VizdoomMyWayHomeTest():

    def __init__(self, env_name, in_channels, batch_size, render):
        self.env_name = env_name
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.render = render

    def run(self, method):
        optional_params = { "show_processed_image": True, "seed": 18 }

        raw_env = gym.make(self.env_name, render_mode="human")
        env = GymEnvVizdoomMyWayHome(raw_env, optional_params)
        worker = SkipAndFramesEnv(env, self.in_channels, (42, 42), optional_params = optional_params)

        agente = Agent(
            method.main_model_factory,
            method.curiosity_model_factory,
            method.save_path
        )
        
        while True:
            done = False
            observation = worker.reset()
            hx = agente.get_new_hx()

            max_reward = 0

            while not done: 
                action, next_hx = agente.get_action_max_prob(observation, hx)

                next_state, reward, done = worker.step(action.item())

                observation = next_state
                hx = next_hx

                if(reward > max_reward):
                    max_reward = reward

                if done:
                    print('Gano' if max_reward > 0.8 else 'Perdio')