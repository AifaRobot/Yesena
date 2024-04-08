import torch

def discount(rewards, dones, gamma):
    value_targets = []
    old_value_target = 0
    
    for t in reversed(range(len(rewards)-1)):
        old_value_target = rewards[t] + gamma*old_value_target*dones[t]
        value_targets.append(old_value_target)
        
    value_targets.reverse()
    return torch.tensor(value_targets)
