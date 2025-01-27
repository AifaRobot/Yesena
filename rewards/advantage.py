import torch
from rewards import Reward
from rewards.utils import discount

class Advantage(Reward):
    def __init__(self, gamma, tau):
        self.gamma = gamma
        self.tau = tau

    '''
        Generalized Advantage Estimation (GAE) is an algorithm used in reinforcement learning to more accurately estimate 
        discounted returns and state values ​​in a Markov decision process (MDP).

        Advantage in reinforcement learning refers to how much better a specific action is compared to other possible 
        actions in a given state. The basic idea behind GAE is to combine the use of multiple time steps. to estimate 
        benefits with a discount factor, similar to how discounted rewards are calculated. This allows for a more stable 
        and efficient estimation of the advantage than taking into account only one step in time.

        The reward for a given time step is not only affected by the current action, but also by all future actions taken. 
        This is because it is possible that even if an action with a good reward is performed at a certain time step, in the 
        future the actions may not return such good rewards, which in the long run would lead to worse performance of the agent. 
        On the contrary, it may happen that performing an action at a certain time returns a bad reward, but those future actions 
        obtain good rewards, which in the long run would lead to better performance of the agent.

        A^GAE_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}

        * A^GAE_t is the estimated advantage at time step t.

        * γ is the discount factor.

        * λ It is a parameter that controls the balance between bias and variance.

        * δ_{t+l} is the temporal error or TD error at time step t+l, calculated as δ_{t+l} = r_{t+l} + γ * V(s_{t+l+1} ) − V(s_{t+l}), 
        where r_{t+l} is the reward at time step t+l and V is the state value function.
    '''

    def calculate_generalized_advantage_estimate(self, rewards, values, next_values, dones):
        delta_t = rewards + self.gamma * next_values * dones - values # 1

        advantages = discount(delta_t, dones, self.gamma*self.tau)

        return advantages

'''
    1. Temporal error is used to estimate how much the actual reward obtained differs at a specific time step. of the 
    expected reward according to the agent's current estimate. It is a fundamental part of the reinforcement. learning 
    algorithms such as TD-learning or Q-learning.

    The general formula for the temporal error at a time step t is:
    
    δ_{t} = r_{t} + γ * V(s_{t+1}) − V(s_{t} )

    Where:

    * δ_{t} is the temporal error at time step t.

    * r_{t} is the reward obtained at time step t.

    * γ is the discount factor.

    * V(s_{t+1}) is the estimate of the value function of the next state s_{t+1}.

    * V(s_{t}) is the estimate of the value function of the current state s_{t}.
''' 