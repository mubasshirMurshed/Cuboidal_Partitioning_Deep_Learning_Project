# Imports
from torch import nn
from torch import Tensor


# Define model
class MNIST_CNN_2(nn.Module):
    """
    A simple CNN for the MNIST dataset with two convolutional layers
    and one linear layer. This is modified for hnadling small resized MNIST images
    """
    def __init__(self):
        """
        Defines the layers of the neural network
        """
        super().__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=0),                              
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(         
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=0),     
            nn.ReLU(),                      
            # nn.MaxPool2d(kernel_size=2),                
        )

        self.lin = nn.Linear(64 * 2 * 2, 10)
        
        
    def forward(self, x: Tensor):
        """
        Defines how the input is propogated through the network.

        Args:
        - x: Tensor
            - A batch of images
        """
        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)   

        output = self.lin(x)
        return output