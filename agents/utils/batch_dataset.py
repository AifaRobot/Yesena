import torch

class Batch_DataSet(torch.utils.data.Dataset):

    def __init__(self, observations = None, next_observations = None, actions = None, advantages = None, 
                    old_log_probs = None, hxs = None, value_targets = None, distributions = None):
        super().__init__()
        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.advantages = advantages
        self.old_log_probs = old_log_probs
        self.hxs = hxs
        self.value_targets = value_targets
        self.distributions = distributions

    def __len__(self):
        return self.observations.shape[0]
    
    def __getitem__(self, i):

        batches = dict()
        
        if (self.observations != None):
            batches['observations_batch'] = self.observations[i] 
        
        if (self.next_observations != None):
            batches['next_observations_batch'] = self.next_observations[i] 

        if (self.actions != None):
            batches['actions_batch'] = self.actions[i] 
        
        if (self.advantages != None):
            batches['advantages_batch'] = self.advantages[i] 

        if (self.old_log_probs != None):
            batches['old_log_probs_batch'] = self.old_log_probs[i] 
    
        if (self.hxs != None):
            batches['hxs_batch'] = self.hxs[i] 

        if (self.value_targets != None):
            batches['value_targets_batch'] = self.value_targets[i]
    
        if (self.distributions != None):
            batches['distributions_batch'] = self.distributions[i] 

        return batches