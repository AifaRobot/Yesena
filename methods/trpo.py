import torch
from methods.utils.utils import *
from torch.utils.data import DataLoader
from agents.utils import Batch_DataSet
from torch.distributions import Categorical
from methods.method_base import MethodBase

'''
    Trust Region Policy Optimization (TRPO) es un algoritmo de aprendizaje por refuerzo diseñado para optimizar políticas en 
    entornos continuos y de gran dimensión, como en robótica o juegos complejos. TRPO se centra en mejorar la estabilidad y 
    eficiencia de las actualizaciones de políticas para evitar cambios drásticos que puedan degradar el rendimiento.
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

        # Se crea la funcion para calcular el producto del Vector Fisher. El Vector Fisher cuantifica cómo cambia la función de costo en 
        # la dirección específica dada por y
        def fisher_vector_product(y):
            kl = loss_fn()["kl"]

            grads = torch.autograd.grad(kl, self.global_agent.actor_critic.actor.parameters(), create_graph=True)
            flat_grads = torch.cat([g.view(-1) for g in grads])

            inner_prod = flat_grads.t() @ y  

            grads = torch.autograd.grad(inner_prod, self.global_agent.actor_critic.actor.parameters())
            flat_grads = torch.cat([g.contiguous().view(-1) for g in grads]).data
            return flat_grads + y * self.damping

        # Calculamos el gradiente conjugado para conocer la direccion de cambio de nuestros pesos y sesgos de la red actor
        opt_dir = conjugate_gradient(fisher_vector_product, -j, self.k) 
        quadratic_term = (opt_dir * fisher_vector_product(opt_dir)).sum()
        # beta es la magnitud del máximo paso posible y se calcula con la siguiente ecuación β = (2 * δ) / (s^T * Hs)
        beta = torch.sqrt(2 * self.trust_region / (quadratic_term + 1e-6))
        # El paso optimo se calcula como la multiplicacion entre beta (la longitud del paso) y opt_dir (la direccion del paso)
        opt_step = beta * opt_dir

        with torch.no_grad():
            old_loss = loss_fn()["actor_loss"]
            # Se obtienen los pesos y sesgos de la red neuronal actor para utilizarlos en caso de que el 
            # procesos de line search no mejore la perdida del actor
            flat_params = get_flat_params_from(self.global_agent.actor_critic.actor) 
            exponent_shrink = 1
            params_updated = False

            for _ in range(self.line_search_num):
                new_params = flat_params + opt_step * exponent_shrink

                # Se reemplazan los pesos y sesgos de la red neuronal actor por los nuevos
                set_flat_params_to(new_params, self.global_agent.actor_critic.actor)
                tmp = loss_fn()
                new_loss = tmp["actor_loss"]
                new_kl = tmp["kl"]

                # Se resta la recompensa vieja con la nueva. Si el resultado es positivo significa que hubo una mejora porque 
                # se redujo el error
                improvement = old_loss - new_loss 

                if new_kl < self.trust_region and improvement >= 0 and torch.isfinite(new_loss):
                    # Si la divergencia de kullback-leibler es es menor a la region de confianza y la nueva perdida es menor a la 
                    # perdida vieja, se termina el bucle y se deja el actor con los nuevos pesos y sesgos
                    params_updated = True
                    break

                exponent_shrink *= 0.5

            if not params_updated:
                # Si los nuevos pesos y sesgos de la red neuronal actor no reducen la funcion de perdida, reemplaza todos los pesos y sesgos 
                # por los que tenia inicialmente
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
