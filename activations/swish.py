import torch.nn as nn

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * nn.Sigmoid()(x)
