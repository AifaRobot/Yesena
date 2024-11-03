import torch
from torch.utils.data import DataLoader
from agents.utils import Batch_DataSet
from torch.distributions import Categorical
from methods.method_base import MethodBase

'''
    PPO significa Optimización de Políticas Próximas, es un algoritmo de aprendizaje por refuerzo (RL) popular y eficaz. 
    Se utiliza para capacitar a agentes de RL en entornos donde las acciones afectan directamente las recompensas que recibe el agente. PPO 
    se centra en optimizar las políticas de decisión del agente de una manera estable y eficiente.

    Una característica clave de PPO es su enfoque en la actualización de las políticas "proximalmente", lo que significa que limita cuánto 
    puede cambiar la política en cada paso de actualización. Esto ayuda a evitar cambios políticos drásticos que podrían ser perjudiciales 
    para el aprendizaje.

    PPO ha demostrado ser eficaz en una variedad de entornos RL y es especialmente popular en aplicaciones de control y juegos. 
    donde se necesita un aprendizaje estable y eficiente.
'''

class PPO(MethodBase):
    def __init__(self, main_model_factory, worker_factory, test_factory, 
            arguments, curiosity_model_factory = '', optimizer = '', generalized_value = '', 
            generalized_advantage = '', save_path = ''):

        self.name = 'PPO'

        super().__init__(arguments, main_model_factory, worker_factory, test_factory, 
            curiosity_model_factory, optimizer, generalized_value, generalized_advantage,
            save_path, self.name)

        self.global_agent.optimizer = self.optimizer(
            list(self.global_agent.curiosity.parameters()) +
            list(self.global_agent.actor_critic.parameters()),
            lr=self.learning_rate
        )


    def calc_loss(self, observations_batch, actions_batch, old_log_probs_batch, value_targets, hxs_batch, advantages_batch, values):    
        distributions, current_values, _ = self.global_agent.actor_critic.forward(observations_batch, hxs_batch)

        m = Categorical(distributions)
        current_log_probs_batch = m.log_prob(actions_batch)     
        entropys = m.entropy()

        current_values = current_values.squeeze(1)

        ratios = torch.exp(current_log_probs_batch - old_log_probs_batch)

        surr1 = ratios * advantages_batch
        surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages_batch

        actor_loss = -torch.min(surr1, surr2)

        if(self.critic_loss_cliping == True):
            CRITIC_LOSS_CLIP = self.critic_loss_clip
            clipped_value_loss = values + torch.clamp(current_values - values, -CRITIC_LOSS_CLIP, CRITIC_LOSS_CLIP)
            v_loss1 = (value_targets - clipped_value_loss) ** 2
            v_loss2 = (value_targets - current_values) ** 2
            critic_loss = torch.max(v_loss1, v_loss2)
        else:
            critic_loss = torch.pow(current_values - value_targets, 2)

        ac_loss = actor_loss.mean() + self.value_coeficient * critic_loss.mean() - self.entropy_coeficient * entropys.mean()

        return ac_loss

    def update(self, rollout):
        actor_critic_losses = []
        curiosity_losses = []

        observations, next_observations, actions, advantages, old_log_probs, hxs, value_targets, distributions, values, extrinsic_rewards, intrinsic_rewards = rollout

        dataset = Batch_DataSet(observations, next_observations, actions, advantages, old_log_probs, hxs, value_targets, distributions, values)
        dataloader = DataLoader(dataset, batch_size=self.minibatch_size, num_workers=0, shuffle=True)
        
        for i, batch in enumerate(dataloader):
            
            for _ in range(self.n_updates * self.num_processes):

                observations_batch = batch['observations_batch']
                next_observations_batch = batch['next_observations_batch']
                actions_batch = batch['actions_batch']
                advantages_batch = batch['advantages_batch']
                old_log_probs_batch = batch['old_log_probs_batch']
                hxs_batch = batch['hxs_batch']
                value_targets_batch = batch['value_targets_batch']
                values_batch = batch['values_batch']

                actor_critic_loss = self.calc_loss(
                    observations_batch, 
                    actions_batch, 
                    old_log_probs_batch, 
                    value_targets_batch, 
                    hxs_batch.squeeze(1), 
                    advantages_batch, 
                    values_batch
                )

                curiosity_loss = self.global_agent.curiosity.calc_loss(observations_batch, next_observations_batch, actions_batch)  

                loss = actor_critic_loss + curiosity_loss

                self.global_agent.optimizer.zero_grad()

                loss.backward()

                self.global_agent.optimizer.step()

                actor_critic_losses.append(actor_critic_loss.item())
                curiosity_losses.append(curiosity_loss.item())

        return dict(actor_critic_losses=actor_critic_losses, curiosity_losses=curiosity_losses, extrinsic_rewards=extrinsic_rewards, intrinsic_rewards=intrinsic_rewards)
