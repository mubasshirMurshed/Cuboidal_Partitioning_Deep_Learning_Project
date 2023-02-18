# Imports
from torch import nn
from torch import Tensor


class MNIST_Linear_2(nn.Module):
    """
    Simple linear fully-connected network for MNIST dataset.
    """
    def __init__(self):
        """
        Initialising a single linear layer.
        """
        super(MNIST_Linear_2, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(16, 10)
        
        
    def forward(self, x: Tensor):
        """
        Defining forward propogation.

        Args:
        - x: Tensor
            - Image data
        """
        x = self.flatten(x)
        x = self.linear1(x)
        return x