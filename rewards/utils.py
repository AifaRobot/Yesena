import torch

'''
    Discounted Rewards are a common technique in reinforcement learning that helps agents learn more effectively 
    in environments where rewards may occur in the future and may be correlated.

    G_{t} = R_{t+1} + γ * R_{t+2} +γʌ2 * R_{t+3} + … 

    * G_{t} it is the reward that is discounted with the passage of time.

    * R_{t+1} is the reward in the next step.

    * 𝛾 is the discount factor. Controls the relative importance of future rewards compared to immediate rewards. 𝛾 
    closer to 1 indicates that future rewards are more important, while an 𝛾 closer to 0 indicates that future rewards 
    are less important relative to immediate rewards.

    * Addition extends to infinity, but in practice it is usually limited to a finite number of future steps.

    The logic behind discounted rewards in reinforcement learning is to capture the concept of "value" in the long term. 
    In environments where actions have long-term consequences and where rewards can be delayed in time, it is important for 
    an agent to consider not only immediate rewards, but also future rewards.

    Reward discounting serves to model the preference for receiving a reward now rather than in the future, due to factors 
    such as uncertainty, the possibility that the agent ceases to exist, or that the environment changes. The idea is that 
    a future reward is “discounted” relative to its temporal delay and associated uncertainty.

    For example, in a game environment, a player might prefer a 10-point reward now to a 15-point reward in the future, due 
    to the uncertainty of whether he or she will actually get the future reward and the benefit of having the reward 
    immediately to use it in the game.

    Discounted rewards allow an agent to consider the long-term consequences of its actions and make decisions that maximize 
    the total expected reward over time, taking into account uncertainty and delay in future rewards.
'''

def discount(rewards, dones, gamma):
    value_targets = []
    old_value_target = 0

    for t in reversed(range(len(rewards))):
        old_value_target = rewards[t] + gamma*old_value_target*dones[t]
        value_targets.append(old_value_target)

    value_targets.reverse()

    return  torch.tensor(value_targets)
