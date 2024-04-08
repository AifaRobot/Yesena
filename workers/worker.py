class Worker:
    def __init__(self, env, agente, batch_size):
        self.env = env
        self.agente = agente
        self.batch_size = batch_size
        self.observation = self.env.reset()

        self.hx = self.agente.get_new_hx()

    def run(self):
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []
        hxs = []
        intrinsic_rewards = []
        distributions = []

        for _ in range(self.batch_size):

            log_prob, value, action, next_hx, distribution = self.agente.get_action(self.observation, self.hx)

            next_observation, reward, done = self.env.step(action.item())

            intrinsic_reward = self.agente.curiosity.calc_reward(self.observation, next_observation, action)

            if done:
                self.hx = self.agente.get_new_hx()
                next_observation = self.env.reset()
            
            observations.append(self.observation.squeeze(0))
            actions.append(action)
            rewards.append(reward)
            dones.append(1 if done == False else 0)
            values.append(value.item())
            log_probs.append(log_prob.detach())
            hxs.append(self.hx)
            intrinsic_rewards.append(intrinsic_reward)
            distributions.append(distribution.detach())

            self.hx = next_hx
            self.observation = next_observation

        _, last_value, _, _, _ = self.agente.get_action(self.observation, self.hx)

        return [observations, actions, rewards, dones, values, log_probs, hxs, intrinsic_rewards, last_value.unsqueeze(0), distributions]
