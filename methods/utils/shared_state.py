import torch
from multiprocessing.managers import SyncManager

'''
    La clase SharedState se encarga de recopilar los datos (observaciones, acciones, valores, etc.) de cada uno de los entornos paralelos  
    para usarlos en el proceso de optimización del agente. 
'''

class SharedState:
    def __init__(self, num_proceses, lock):
        
        self.lock = lock
        self.num_proceses = num_proceses

        manager = SyncManager()
        manager.start()

        '''
        Se crean las listas vacías donde van a guardarse los datos necesarios para el proceso de optimización del agente. Esos datos son: 
        
        *shared_observations: Aqui se guardarán las observaciones recolectadas por el agente.
        
        *shared_next_observations: Aquí se guardarán las siguientes observaciones, que serían la siguiente observación para cada una 
            de las observaciones que hay en shared_observations. shared_next_observations solo es útil en caso de que el agente use  
            curiosity.
        
        *shared_actions: Aquí se guardarán las acciones que realizo el agente en cada uno de los estados alojados en shared_observations. 
        
        *shared_advantages: Aquí se guardarán las ventajas. Las ventajas son una medida de que tan mala fueron las acciones tomadas en  
            cada uno de los estados alojados en shared_observations.
        
        *shared_old_log_probs: Aquí se guardarán los logaritmos de las probabilidades de realizar cada una de las acciones en cada uno 
            de los estados en alojados en shared_observations.
        
        *shared_hxs: Aquí se guardarán los estados ocultos. Los estados ocultos son usados por algunas redes neuronales que requieren  
            una memoria para trabajar en algunos entornos.
        
        *shared_value_targets: Aquí se guardarán los valores que tiene cada observacion en shared_observations y que debe aprender el crítico.
        
        *shared_distributions: Aquí se guardarán las distribuciones que el actor produjo para cada una de las observaciones alojadas en shared_observations.
        
        *shared_values: Aquí se guardarán los valores que el crítico produjo para cada una de las observaciones alojadas en shared_observations.
        
        *shared_extrinsic_rewards: Son las recompensas obtenidas desde el entorno al realizar las acciones alojadas en shared_actions en cada 
            uno de los estados alojados en shared_observations. 
        
        *shared_intrinsic_rewards: Aquí se guardarán las recompensas generadas por el propio agente. shared_next_observations solo es útil en caso de que el agente use  
                curiosity. 
        '''
        self.shared_observations = manager.list()
        self.shared_next_observations = manager.list()
        self.shared_actions = manager.list()
        self.shared_advantages = manager.list()
        self.shared_old_log_probs = manager.list()
        self.shared_hxs = manager.list()
        self.shared_value_targets = manager.list()
        self.shared_distributions = manager.list()
        self.shared_values = manager.list()
        self.shared_extrinsic_rewards = manager.list()
        self.shared_intrinsic_rewards = manager.list()

    def update_agent(self, rollout, method, cicle):
        observations_batch, actions_batch, extrinsic_rewards_batch, dones_batch, values_batch, old_log_probs_batch, hxs_batch, intrinsic_rewards_batch, distributions_batch = rollout

        observations = torch.stack(observations_batch[:-1])
        next_observations = torch.stack(observations_batch[1:])
        distributions = torch.stack(distributions_batch[:-1])
        actions = torch.stack(actions_batch[:-1])
        old_log_probs = torch.stack(old_log_probs_batch[:-1])
        hxs = torch.stack(hxs_batch[:-1])
        values = torch.tensor(values_batch[:-1])
        next_values = torch.tensor(values_batch[1:])
        dones = torch.tensor(dones_batch[:-1])
        extrinsic_rewards = torch.tensor(extrinsic_rewards_batch[:-1])
        intrinsic_rewards = torch.tensor(intrinsic_rewards_batch[:-1])
        rewards = extrinsic_rewards + intrinsic_rewards
        advantages = method.generalized_advantage.calculate_generalized_advantage_estimate(rewards, values, next_values, dones)
        value_targets = method.generalized_value.calculate_discounted_rewards(rewards, dones)

        if(method.advantages_normalize == True):
           advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-8)

        if(method.value_targets_normalize == True):
           value_targets = (value_targets - torch.mean(value_targets)) / (torch.std(value_targets) + 1e-8)

        with self.lock:
            self.append_memory(
                observations,
                next_observations,
                actions,
                advantages,
                old_log_probs,
                hxs,
                value_targets,
                distributions,
                values,
                extrinsic_rewards,
                intrinsic_rewards,
            )

            if(len(self.shared_observations) == self.num_proceses):
                temporary_observations = []
                temporary_next_observations = []
                temporary_actions = []
                temporary_advantages = []
                temporary_old_log_probs = []
                temporary_hxs = []
                temporary_value_targets = []
                temporary_distributions = []
                temporary_values = []
                temporary_extrinsic_rewards = []
                temporary_intrinsic_rewards = []

                for (
                    local_observations,
                    local_next_observations, 
                    local_actions, 
                    local_advantages, 
                    local_old_log_probs, 
                    local_hxs, 
                    local_value_targets, 
                    local_distributions, 
                    local_values, 
                    local_extrinsic_rewards, 
                    local_intrinsic_rewards
                ) in zip (
                    self.shared_observations
                    ,self.shared_next_observations
                    ,self.shared_actions
                    ,self.shared_advantages
                    ,self.shared_old_log_probs
                    ,self.shared_hxs
                    ,self.shared_value_targets
                    ,self.shared_distributions
                    ,self.shared_values
                    ,self.shared_extrinsic_rewards
                    ,self.shared_intrinsic_rewards
                ):
                    temporary_observations = local_observations if temporary_observations == [] else torch.cat((temporary_observations, local_observations), dim=0)
                    temporary_next_observations = local_next_observations if temporary_next_observations == [] else torch.cat((temporary_next_observations, local_next_observations), dim=0)
                    temporary_actions = local_actions if temporary_actions == [] else torch.cat((temporary_actions, local_actions), dim=0)
                    temporary_advantages = local_advantages if temporary_advantages == [] else torch.cat((temporary_advantages, local_advantages), dim=0)
                    temporary_old_log_probs = local_old_log_probs if temporary_old_log_probs == [] else torch.cat((temporary_old_log_probs, local_old_log_probs), dim=0)
                    temporary_hxs = local_hxs if temporary_hxs == [] else torch.cat((temporary_hxs, local_hxs), dim=0)
                    temporary_value_targets = local_value_targets if temporary_value_targets == [] else torch.cat((temporary_value_targets, local_value_targets), dim=0)
                    temporary_distributions = local_distributions if temporary_distributions == [] else torch.cat((temporary_distributions, local_distributions), dim=0)
                    temporary_values = local_values if temporary_values == [] else torch.cat((temporary_values, local_values), dim=0)
                    temporary_extrinsic_rewards = local_extrinsic_rewards if temporary_extrinsic_rewards == [] else torch.cat((temporary_extrinsic_rewards, local_extrinsic_rewards), dim=0)
                    temporary_intrinsic_rewards = local_intrinsic_rewards if temporary_intrinsic_rewards == [] else torch.cat((temporary_intrinsic_rewards, local_intrinsic_rewards), dim=0)

                dict_metrics = method.update((
                    temporary_observations, 
                    temporary_next_observations, 
                    temporary_actions, 
                    temporary_advantages, 
                    temporary_old_log_probs, 
                    temporary_hxs, 
                    temporary_value_targets, 
                    temporary_distributions, 
                    temporary_values, 
                    temporary_extrinsic_rewards, 
                    temporary_intrinsic_rewards
                ))

                self.clean_memory()

                method.update_metrics(cicle, dict_metrics)

                if(cicle == method.episodes-1):
                    method.save_models()
                    method.draw_plots()

    '''
        Agrega a la memoria compartida los siguientes datos:
        * Observaciones (observations)
        * Siguientes Observaciones (next_observations)
        * Acciones (actions)
        * Ventajas (advantages)
        * Probabilidades Logaritmicas (old_log_probs)
        * Estados Ocultos (hxs)
        * Valores Objetivos (value_targets)
        * Distribuciones (distributions)
        * Valores (vales)
        * Recompensas Extrinsecas (extrinsic_rewards)
        * Recompensas Intrinsecas (intrinsic_rewards)
    '''
    def append_memory(self, observations, next_observations, actions, advantages, old_log_probs, hxs, value_targets, distributions, values, extrinsic_rewards, intrinsic_rewards):
        self.shared_observations.append(observations)
        self.shared_next_observations.append(next_observations)
        self.shared_actions.append(actions)
        self.shared_advantages.append(advantages)
        self.shared_old_log_probs.append(old_log_probs)
        self.shared_hxs.append(hxs.detach())
        self.shared_value_targets.append(value_targets)
        self.shared_distributions.append(distributions)
        self.shared_values.append(values.detach())
        self.shared_extrinsic_rewards.append(extrinsic_rewards)
        self.shared_intrinsic_rewards.append(intrinsic_rewards)

    '''
        Elimina los datos en memora cuando ya fueron utilizados en el proceso de optimización del agente.
    '''
    def clean_memory(self):
        self.shared_observations[:] = []
        self.shared_next_observations[:] = []
        self.shared_actions[:] = []
        self.shared_advantages[:] = []
        self.shared_old_log_probs[:] = []
        self.shared_hxs[:] = []
        self.shared_value_targets[:] = []
        self.shared_distributions[:] = []
        self.shared_values[:] = []
        self.shared_extrinsic_rewards[:] = []
        self.shared_intrinsic_rewards[:] = []
