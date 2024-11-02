import torch

'''
    The batch_datase class is used to store all the data necessary for training the agent. 

    Usually the agent is not trained with all the data but rather it is divided into smaller batches 
    (minibatches) to reduce memory requirements and allow the data to be processed in a more manageable way.

    The information saved is the following:

    * Observations: All states the agent was in.
    * Next Observations: The next state for each state the agent was in.
    * Actions: They are the actions that the agent has taken in each state.
    * Advantages: They are the difference between the reward for actions taken and the estimated value for those actions.
    * Old_log_probs: They are the logarithm of the probability of each action chosen in each state.
    * Hxs: They are the hidden state of each state in which the agent was. These are only useful when neural networks 
        with their own memory are used.
    * Value_targets: They are the correct values ​​that the neural network should return for each state.
    * Distributions: They are the probability distributions of taking each of the possible actions for the agent for each 
        state in which the agent was.
    * Values: These are the values ​​that the agent estimated for each state.
'''


class Batch_DataSet(torch.utils.data.Dataset):

    def __init__(self, observations = None, next_observations = None, actions = None, advantages = None, 
                    old_log_probs = None, hxs = None, value_targets = None, distributions = None, values = None):
        super().__init__()
        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.advantages = advantages
        self.old_log_probs = old_log_probs
        self.hxs = hxs
        self.value_targets = value_targets
        self.distributions = distributions
        self.values = values

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

        if (self.values != None):
            batches['values_batch'] = self.values[i] 

        return batches