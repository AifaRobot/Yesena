import torch
from methods.utils.utils import *
from torch.utils.data import DataLoader
from agents.utils import Batch_DataSet
from torch.distributions import Categorical
from methods.method_base import MethodBase

'''
    Trust Region Policy Optimization (TRPO) is a reinforcement learning algorithm designed to optimize policies 
    in continuous, high-dimensional environments, such as in robotics or complex games. TRPO focuses on improving 
    the stability and efficiency of policy updates to avoid drastic changes that can degrade performance.
'''

class TRPO(MethodBase):
    def __init__(self, main_model_factory, worker_factory, test_factory, 
            arguments, curiosity_model_factory = '', optimizer = '', generalized_value = '', 
            generalized_advantage = '', save_path = ''):
        
        self.name = "TRPO"

        super().__init__(arguments, main_model_factory, 
            worker_factory, test_factory, curiosity_model_factory, 
            optimizer, generalized_value, generalized_advantage,
            save_path, self.name)

        self.global_agent.optimizer = self.optimizer(
            list(self.global_agent.curiosity.parameters()) +
            list(self.global_agent.actor_critic.critic.parameters()),
            lr=self.learning_rate
        )

    def optimize_actor(self, loss_dict, loss_fn):
        loss = loss_dict["actor_loss"]
        grads = torch.autograd.grad(loss, self.global_agent.actor_critic.actor.parameters())
        j = torch.cat([g.view(-1) for g in grads]).data

        # The function is created to calculate the product of the Fisher Vector. The Fisher 
        # Vector quantifies how the cost function changes in the specific direction given by y
        def fisher_vector_product(y):
            kl = loss_fn()["kl"]

            grads = torch.autograd.grad(kl, self.global_agent.actor_critic.actor.parameters(), create_graph=True)
            flat_grads = torch.cat([g.view(-1) for g in grads])

            inner_prod = flat_grads.t() @ y  

            grads = torch.autograd.grad(inner_prod, self.global_agent.actor_critic.actor.parameters())
            flat_grads = torch.cat([g.contiguous().view(-1) for g in grads]).data
            return flat_grads + y * self.damping

        # We calculate the conjugate gradient to know the direction of change of our weights and biases of the actor network
        opt_dir = conjugate_gradient(fisher_vector_product, -j, self.k) 
        quadratic_term = (opt_dir * fisher_vector_product(opt_dir)).sum()
        # beta is the magnitude of the maximum possible step and is calculated with the following equation β = (2 * δ) / (s^T * Hs)
        beta = torch.sqrt(2 * self.trust_region / (quadratic_term + 1e-6))
        # The optimal step is calculated as the multiplication between beta (the length of the step) and opt_dir (the direction of the step)
        opt_step = beta * opt_dir

        with torch.no_grad():
            old_loss = loss_fn()["actor_loss"]
            # The weights and biases of the actor neural network are obtained to be used in case the line 
            # search process does not improve the actor's loss
            flat_params = get_flat_params_from(self.global_agent.actor_critic.actor) 
            exponent_shrink = 1
            params_updated = False

            for _ in range(self.line_search_num):
                new_params = flat_params + opt_step * exponent_shrink

                # The weights and biases of the actor neural network are replaced with the new ones
                set_flat_params_to(new_params, self.global_agent.actor_critic.actor)
                tmp = loss_fn()
                new_loss = tmp["actor_loss"]
                new_kl = tmp["kl"]

                # The old reward is subtracted from the new one. If the result is positive it means that there 
                # was an improvement because the error was reduced
                improvement = old_loss - new_loss 

                if new_kl < self.trust_region and improvement >= 0 and torch.isfinite(new_loss):
                    # If the kullback-leibler divergence is less than the confidence region and the new loss is 
                    # less than the old loss, the loop ends and the actor is left with the new weights and biases
                    params_updated = True
                    break

                exponent_shrink *= 0.5

            if not params_updated:
                # If the new weights and biases of the actor neural network do not reduce the loss function, 
                # replace all the weights and biases with the ones it initially had
                set_flat_params_to(flat_params, self.global_agent.actor_critic.actor)

    def trpo_step(self, observations, actions, old_log_probs, old_distributions, advantages):

        def get_actor_loss():
            distributions = self.global_agent.actor_critic.actor.forward(observations)
 
            m = Categorical(distributions)
            log_prob = m.log_prob(actions)
            entropys = m.entropy()

            ratios = torch.exp(log_prob - old_log_probs)

            actor_loss = -torch.mean(ratios * advantages) - self.entropy_coeficient * entropys.mean()
            kl = categorical_kl(distributions, old_distributions).mean() # Se calcula la divergencia de kullback-leibler

            return dict(actor_loss=actor_loss, kl=kl)

        loss_dict = get_actor_loss()

        self.optimize_actor(loss_dict, get_actor_loss)

        return loss_dict['actor_loss']

    def update(self, rollout):
        actor_losses = []
        critic_losses = []
        curiosity_losses = []

        observations, next_observations, actions, advantages, old_log_probs, hxs, value_targets, distributions, values, extrinsic_rewards, intrinsic_rewards = rollout

        actor_loss = self.trpo_step(observations, actions, old_log_probs, distributions, advantages)

        actor_losses.append(actor_loss.item())

        dataset = Batch_DataSet(observations = observations, value_targets = value_targets, next_observations = next_observations, actions = actions)
        dataloader = DataLoader(dataset, batch_size=self.minibatch_size, num_workers=0, shuffle=True)

        for _, batch in enumerate(dataloader):

            for _ in range(self.num_processes * self.n_updates):

                observations_batch = batch['observations_batch']
                value_targets_batch = batch['value_targets_batch']
                next_observations_batch = batch['next_observations_batch']
                actions_batch = batch['actions_batch']

                curiosity_loss = self.global_agent.curiosity.calc_loss(observations_batch, next_observations_batch, actions_batch)

                current_values = self.global_agent.actor_critic.critic.forward(observations_batch)

                current_values = current_values.squeeze(1)

                critic_loss = self.value_coeficient * torch.pow(current_values - value_targets_batch, 2).mean()

                total_loss = curiosity_loss + critic_loss
                
                self.global_agent.optimizer.zero_grad()

                total_loss.backward()

                self.global_agent.optimizer.step()

                critic_losses.append(critic_loss.item())
                curiosity_losses.append(curiosity_loss.item())

        return dict(actor_losses=actor_losses, critic_losses=critic_losses, curiosity_losses=curiosity_losses, extrinsic_rewards=extrinsic_rewards, intrinsic_rewards=intrinsic_rewards)
