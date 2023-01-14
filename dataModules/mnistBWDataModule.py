from dataModules.dataModule import DataModule
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
from torch import Tensor


class MNISTBWDataModule(DataModule):
    """
    A data module for the normal MNIST dataset
    """
    def __init__(self, train_dir: str, val_dir: str, batch_size: int):
        """
        Save attributes.

        Args:
        - train_dir: str
            - Directory of training dataset
        - val_dir: str
            - Directory of validation dataset
        - batch_size: int
            - How many data samples per batch to be loaded
        """
        super().__init__(train_dir, val_dir, batch_size, DataLoader)
    

    def setup(self):
        """
        Instantiate datasets for training and validation.
        """
        transform = transforms.Compose([transforms.ToTensor(), BWMask(0.2)])

        self.train_set = MNIST(root=self.train_dir, train=True, transform=transform)

        self.val_set = MNIST(root=self.val_dir, train=False, transform=transform)


class BWMask():
    """Converts greyscale to black and white"""
    def __init__(self, threshold: float):
        self.threshold = threshold


    def __call__(self, sample: Tensor):
        """
        Converts values in a given tensor to 1 if they are above a certain
        threshold, otherwise 0.

        Args:
        - sample: Tensor
            - A tensor image
        """
        return torch.where(sample > self.threshold, 1.0, 0.0)