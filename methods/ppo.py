import torch
from torch.utils.data import DataLoader
from agents.utils import Batch_DataSet
from torch.distributions import Categorical
from methods.method_base import MethodBase

class PPO(MethodBase):
    def __init__(self, env_name, main_model_factory, worker_factory, test_factory, 
            arguments, curiosity_model_factory = '', optimizer = '', generalized_value = '', 
            generalized_advantage = '', save_path = ''):

        self.name = 'PPO'

        super().__init__(arguments, main_model_factory, worker_factory, test_factory, 
            curiosity_model_factory, optimizer, generalized_value, generalized_advantage,
            save_path, env_name, self.name)

        self.global_agent.actor_critic.share_memory()
        self.global_agent.curiosity.share_memory()

        self.global_agent.optimizer = self.optimizer(
            list(self.global_agent.curiosity.parameters()) +
            list(self.global_agent.actor_critic.parameters()),
            lr=self.learning_rate
        )

    def calc_loss(self, observations_batch, actions_batch, old_log_probs_batch, value_targets, hxs_batch, advantages_batch, local_actor_critic):
        distributions, current_values, _ = local_actor_critic.forward(observations_batch, hxs_batch)

        m = Categorical(distributions)
        current_log_probs_batch = m.log_prob(actions_batch)     
        entropys = m.entropy()

        current_values = current_values.squeeze(1)
                
        ratios = torch.exp(current_log_probs_batch - old_log_probs_batch)

        surr1 = ratios * advantages_batch
        surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages_batch

        actor_loss = -torch.min(surr1, surr2)

        critic_loss = torch.pow(current_values - value_targets, 2)

        ac_loss = actor_loss.mean() + self.c1 * critic_loss.mean() - self.c2 * entropys.mean()

        return ac_loss

    def update(self, rollout, local_agent):
        actor_critic_losses = []
        curiosity_losses = []

        observations_batch, actions_batch, extrinsic_rewards_batch, dones_batch, values_batch, old_log_probs_batch, hxs_batch, intrinsic_rewards_batch, last_value, distributions_batch = rollout

        observations = torch.stack(observations_batch[:-1])
        next_observations = torch.stack(observations_batch[1:])
        distributions = torch.stack(distributions_batch[:-1])
        actions = torch.stack(actions_batch[:-1])
        old_log_probs = torch.stack(old_log_probs_batch[:-1])
        hxs = torch.stack(hxs_batch[:-1]).detach_()
        values = torch.tensor(values_batch)
        dones = torch.tensor(dones_batch)

        extrinsic_rewards = torch.tensor(extrinsic_rewards_batch)
        intrinsic_rewards = torch.tensor(intrinsic_rewards_batch)

        combine_rewards = extrinsic_rewards + intrinsic_rewards        
        
        values = torch.cat((values, last_value), dim=0)
        advantages = self.generalized_advantage.calculate_generalized_advantage_estimate(combine_rewards, values, dones)
        
        combine_rewards = torch.cat((combine_rewards, last_value), dim=0)
        value_targets = self.generalized_value.calculate_discounted_rewards(combine_rewards, dones)[:-1]

        dataset = Batch_DataSet(observations, next_observations, actions, advantages, old_log_probs, hxs, value_targets, distributions)
        dataloader = DataLoader(dataset, batch_size=self.minibatch_size, num_workers=0, shuffle=True)

        for _ in range(self.n_updates):
                
            for _, batch in enumerate(dataloader):                        
                observations_batch = batch['observations_batch']
                next_observations_batch = batch['next_observations_batch']
                actions_batch = batch['actions_batch']
                advantages_batch = batch['advantages_batch']
                old_log_probs_batch = batch['old_log_probs_batch']
                hxs_batch = batch['hxs_batch']
                value_targets_batch = batch['value_targets_batch']

                actor_critic_loss = self.calc_loss(
                    observations_batch, 
                    actions_batch, 
                    old_log_probs_batch, 
                    value_targets_batch, 
                    hxs_batch, 
                    advantages_batch, 
                    local_agent.actor_critic
                )

                curiosity_loss = local_agent.curiosity.calc_loss(observations_batch, next_observations_batch, actions_batch)  

                loss = actor_critic_loss + curiosity_loss
                
                self.global_agent.optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.global_agent.actor_critic.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.global_agent.curiosity.parameters(), 0.5)

                for local_param, global_param in zip(local_agent.actor_critic.parameters(), self.global_agent.actor_critic.parameters()):
                        global_param._grad = local_param.grad

                for local_param, global_param in zip(local_agent.curiosity.parameters(), self.global_agent.curiosity.parameters()):
                        global_param._grad = local_param.grad

                self.global_agent.optimizer.step()

                local_agent.actor_critic.load_state_dict(self.global_agent.actor_critic.state_dict())
                local_agent.curiosity.load_state_dict(self.global_agent.curiosity.state_dict())

                actor_critic_losses.append(actor_critic_loss.item())
                curiosity_losses.append(curiosity_loss.item())
        
        return dict(actor_critic_losses=actor_critic_losses, curiosity_losses=curiosity_losses, extrinsic_rewards=extrinsic_rewards, intrinsic_rewards=intrinsic_rewards)


