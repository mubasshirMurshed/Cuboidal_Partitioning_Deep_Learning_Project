# Imports
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

class DimensionNetwork(nn.Module):
    """
    A fully connected linear network of dim-2-1.
    """
    def __init__(self, dim: int):
        """
        Defines the linear layers of the network.

        Args:
        - dim: int
            - The input vector length
        """
        super(DimensionNetwork, self).__init__()
        self.linear1 = nn.Linear(dim, 2)
        self.linear2 = nn.Linear(2, 1)
        
    def forward(self, x: Tensor):
        """
        Defines forward propogation of the network.

        Args:
        - x: Tensor
            - The input cuboid feature vector
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        return x

class CuboidalSparseNetwork(nn.Module):
    """
    The larger scale mode of the sparse network that has sparse sub-networks for
    each block which then all combines into a fully connected linear network.
    """
    def __init__(self, numCuboids: int, numFeatures: int, numClasses: int):
        """
        Defines the layers of the network and each sub-network in a ModuleList.

        Args:
        - numCuboids: int
            - The number of cuboids in the partitioning
        - numFeatures: int
            - The number of features associated with each cuboid
        - numClasses: int
            - Number of classes to recognise
        """
        super(CuboidalSparseNetwork, self).__init__()
        self.networks = nn.ModuleList()
        self.numCuboids = numCuboids
        self.numClasses = numClasses
        
        for i in range(self.numCuboids):
            self.networks.append(DimensionNetwork(numFeatures))
        
        self.lin1 = nn.Linear(self.numCuboids, self.numClasses)
        
    def forward(self, image: Tensor):
        """
        Defines forward propogation of the network.

        Args:
        - image: Tensor
            - The input partitioning data
        """
        # Create space for the output
        firstLayerOutput = [0]*self.numCuboids
        for i in range(self.numCuboids):
            inp = image[:, i]   # Get feature vector of cuboid i
            result = self.networks[i](inp) # Pass through feature vector in sub-network i for cuboid i
            firstLayerOutput[i] = result # Store result

        x = torch.cat(firstLayerOutput, 1)  # Concatenate all results together to get a normal tensor of (batch, numCuboids)

        x = self.lin1(x)
        return x