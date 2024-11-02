import torch
from multiprocessing.managers import SyncManager

class SharedState:
    def __init__(self, num_proceses, lock):
        
        self.lock = lock
        self.num_proceses = num_proceses

        manager = SyncManager()
        manager.start()

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
        observations_batch, actions_batch, extrinsic_rewards_batch, dones_batch, values_batch, old_log_probs_batch, hxs_batch, intrinsic_rewards_batch, last_value, distributions_batch = rollout

        observations = torch.stack(observations_batch[:-1])
        next_observations = torch.stack(observations_batch[1:])
        distributions = torch.stack(distributions_batch[:-1])
        actions = torch.stack(actions_batch[:-1])
        old_log_probs = torch.stack(old_log_probs_batch[:-1])
        hxs = torch.stack(hxs_batch[:-1])
        values = torch.tensor(values_batch)
        dones = torch.tensor(dones_batch)
        extrinsic_rewards = torch.tensor(extrinsic_rewards_batch)
        intrinsic_rewards = torch.tensor(intrinsic_rewards_batch)

        rewards = extrinsic_rewards + intrinsic_rewards

        values = torch.cat((values, last_value), dim=0)

        advantages = method.generalized_advantage.calculate_generalized_advantage_estimate(rewards, values, dones)

        value_targets = advantages + values[:-2].detach()

        #rewards = torch.cat((rewards, last_value), dim=0)

        #value_targets = method.generalized_value.calculate_discounted_rewards(rewards, dones)[:-1]

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
