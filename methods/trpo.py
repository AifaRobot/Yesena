import torch
from methods.utils.utils import *
from torch.utils.data import DataLoader
from agents.utils import Batch_DataSet
from torch.distributions import Categorical
from methods.method_base import MethodBase

class TRPO(MethodBase):
    def __init__(self, env_name, main_model_factory, worker_factory, test_factory, 
            arguments, curiosity_model_factory = '', optimizer = '', generalized_value = '', 
            generalized_advantage = '', save_path = ''):
        
        self.name = "TRPO"

        super().__init__(arguments, main_model_factory, 
            worker_factory, test_factory, curiosity_model_factory, 
            optimizer, generalized_value, generalized_advantage,
            save_path, env_name, self.name)

        self.global_agent.curiosity.share_memory()
        self.global_agent.actor_critic.critic.share_memory()

        self.global_agent.optimizer = self.optimizer(
            list(self.global_agent.curiosity.parameters()) +
            list(self.global_agent.actor_critic.critic.parameters()),
            lr=self.learning_rate
        )

    def optimize_actor(self, loss_dict: dict, loss_fn, actor_critic):
        loss = loss_dict["a_loss"]
        grads = torch.autograd.grad(loss, actor_critic.actor.parameters())
        j = torch.cat([g.view(-1) for g in grads]).data

        def fisher_vector_product(y):
            kl = loss_fn()["kl"]

            grads = torch.autograd.grad(kl, actor_critic.actor.parameters(), create_graph=True)
            flat_grads = torch.cat([g.view(-1) for g in grads])

            inner_prod = flat_grads.t() @ y  
            
            grads = torch.autograd.grad(inner_prod, actor_critic.actor.parameters()) 
            flat_grads = torch.cat([g.contiguous().view(-1) for g in grads]).data
            return flat_grads + y * self.damping

        opt_dir = cg(fisher_vector_product, -j, self.k)
        quadratic_term = (opt_dir * fisher_vector_product(opt_dir)).sum()
        beta = torch.sqrt(2 * self.trust_region / (quadratic_term + 1e-6))
        opt_step = beta * opt_dir

        with torch.no_grad():
            old_loss = loss_fn()["a_loss"]
            flat_params = get_flat_params_from(actor_critic.actor)
            exponent_shrink = 1
            params_updated = False

            for _ in range(self.line_search_num):
                new_params = flat_params + opt_step * exponent_shrink

                set_flat_params_to(new_params, actor_critic.actor)
                tmp = loss_fn()
                new_loss = tmp["a_loss"]
                new_kl = tmp["kl"]
                improvement = old_loss - new_loss

                if new_kl < 1.5 * self.trust_region and improvement >= 0 and torch.isfinite(new_loss):
                    params_updated = True
                    return opt_step * exponent_shrink

                exponent_shrink *= 0.5
            if not params_updated:
                set_flat_params_to(flat_params, actor_critic.actor)
            
            return []

    def trpo_step(self, states, actions, old_log_prob, old_probs, advs, actor_critic):

        def get_actor_loss() -> dict:
            probs = actor_critic.actor.forward(states)
 
            dist = Categorical(probs)
            log_prob = dist.log_prob(actions)
            ent = dist.entropy().mean()

            a_loss = -torch.mean((log_prob - old_log_prob).exp() * advs) - self.c2 * ent
            kl = categorical_kl(probs, old_probs).mean()

            return dict(a_loss=a_loss, ent=ent, kl=kl)

        loss_dict = get_actor_loss()

        grads = self.optimize_actor(loss_dict, get_actor_loss, actor_critic)

        return grads, loss_dict['a_loss']

    def update(self, rollout, local_agent):
        actor_losses = []
        critic_losses = []
        curiosity_losses = []

        observations_batch, actions_batch, extrinsic_rewards_batch, dones_batch, values_batch, old_log_probs_batch, hxs_batch, intrinsic_rewards_batch, last_value, distributions_batch = rollout
        
        observations = torch.stack(observations_batch[:-1])
        next_observations = torch.stack(observations_batch[1:])
        actions = torch.stack(actions_batch[:-1])
        old_log_probs = torch.stack(old_log_probs_batch[:-1])
        hxs = torch.stack(hxs_batch[:-1])
        distributions = torch.stack(distributions_batch[:-1])
        values = torch.tensor(values_batch)
        dones = torch.tensor(dones_batch)

        extrinsic_rewards = torch.tensor(extrinsic_rewards_batch)
        intrinsic_rewards = torch.tensor(intrinsic_rewards_batch)

        combine_rewards = extrinsic_rewards + intrinsic_rewards        
        
        values = torch.cat((values, last_value), dim=0)
        advantages = self.generalized_advantage.calculate_generalized_advantage_estimate(combine_rewards, values, dones)
        
        combine_rewards = torch.cat((combine_rewards, last_value), dim=0)
        value_targets = self.generalized_value.calculate_discounted_rewards(combine_rewards, dones)[:-1]
        
        actor_gradients, a_loss = self.trpo_step(
            observations, 
            actions, 
            old_log_probs, 
            distributions, 
            advantages, 
            local_agent.actor_critic
        )

        if(actor_gradients != []):
            flat_params = get_flat_params_from(self.global_agent.actor_critic.actor)
            new_params = actor_gradients + flat_params
            set_flat_params_to(new_params, self.global_agent.actor_critic.actor)

        local_agent.actor_critic.actor.load_state_dict(self.global_agent.actor_critic.actor.state_dict())

        actor_losses.append(a_loss.item())

        dataset = Batch_DataSet(observations = observations, value_targets = value_targets, hxs = hxs, next_observations = next_observations, actions = actions)
        dataloader = DataLoader(dataset, batch_size=self.minibatch_size, num_workers=0, shuffle=True)
        
        for _, batch in enumerate(dataloader):
            for _ in range(self.n_updates):
                observations_batch = batch['observations_batch']
                value_targets_batch = batch['value_targets_batch']
                hxs_batch = batch['hxs_batch']
                next_observations_batch = batch['next_observations_batch']
                actions_batch = batch['actions_batch']

                curiosity_loss = local_agent.curiosity.calc_loss(observations_batch, next_observations_batch, actions_batch)  

                current_values, _ = local_agent.actor_critic.critic.forward(observations_batch, hxs_batch)

                critic_loss = self.c1 * torch.pow(current_values.squeeze(1) - value_targets_batch.detach(), 2).mean()

                total_loss = curiosity_loss + critic_loss
                
                self.global_agent.optimizer.zero_grad()

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.global_agent.actor_critic.critic.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.global_agent.curiosity.parameters(), 0.5)

                for local_param, global_param in zip(local_agent.actor_critic.critic.parameters(), self.global_agent.actor_critic.critic.parameters()):
                    global_param._grad = local_param.grad

                for local_param, global_param in zip(local_agent.curiosity.parameters(), self.global_agent.curiosity.parameters()):
                    global_param._grad = local_param.grad

                self.global_agent.optimizer.step()
                
                local_agent.actor_critic.critic.load_state_dict(self.global_agent.actor_critic.critic.state_dict())
                local_agent.curiosity.load_state_dict(self.global_agent.curiosity.state_dict())

                critic_losses.append(critic_loss.item())
                curiosity_losses.append(curiosity_loss.item())

        return dict(actor_losses=actor_losses, critic_losses=critic_losses, curiosity_losses=curiosity_losses, extrinsic_rewards=extrinsic_rewards, intrinsic_rewards=intrinsic_rewards)
