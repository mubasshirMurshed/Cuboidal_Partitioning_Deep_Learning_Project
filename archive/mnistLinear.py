# Imports
from torch import nn
from torch import Tensor


class MNIST_Linear(nn.Module):
    """
    Simple linear fully-connected network for MNIST dataset.
    """
    def __init__(self):
        """
        Initialising a single linear layer.
        """
        super().__init__()
        self.linear1 = nn.Linear(784, 10)
        
        
    def forward(self, x: Tensor):
        """
        Defining forward propogation.

        Args:
        - x: Tensor
            - Image data
        """
        x = self.linear1(x)
        return x