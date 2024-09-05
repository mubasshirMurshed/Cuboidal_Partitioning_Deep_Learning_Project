from torch.utils.data import Subset, Dataset
from torchvision.datasets import MNIST, CIFAR10, DatasetFolder
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from medmnist import OrganAMNIST
import os
import numpy as np
from typing import Callable, Dict
from abc import ABC, abstractmethod

# TODO: Add k-fold functionality
# TODO: Fix abs paths

# Root directories to dataset sources
MNIST_ROOT = r"D:\Python\Cuboidal_Partitioning_Deep_Learning_Project\data\source\MNIST"
CIFAR10_ROOT = r"D:\Python\Cuboidal_Partitioning_Deep_Learning_Project\data\source\CIFAR10"
MEDMNIST_ROOT = r"D:\Python\Cuboidal_Partitioning_Deep_Learning_Project\data\source\MedMNIST"
OMNIGLOT_ROOT = r"D:\Python\Cuboidal_Partitioning_Deep_Learning_Project\data\source\Omniglot"


class SourceDataset(ABC):
    """
    Base Dataset class to encapsulate links to training, validation and test datasets.
    """
    @staticmethod
    @abstractmethod
    def name() -> str:
        ...

    @abstractmethod
    def train_dataset(self) -> Dataset:
        ...

    @abstractmethod
    def validation_dataset(self) -> Dataset:
        ...

    @abstractmethod
    def test_dataset(self) -> Dataset:
        ...


class MyMNIST(SourceDataset):
    """
    Wrapper of the MNIST dataset.
    """
    @staticmethod
    def name() -> str:
        return "MNIST"

    def __init__(self, transform: Callable | None=None) -> None:
        """
        Sets up the training, validation, and testing dataset of MNIST.
        """
        self.transform = transform
        self.shape = (28, 28, 1)
        
    def train_dataset(self) -> Dataset:
        ds = MNIST(root=MNIST_ROOT, train=True, transform=self.transform)
        train_idx = np.load(r"D:\Python\Cuboidal_Partitioning_Deep_Learning_Project\data\split_indices\MNIST_Train_Idx.npy")
        train_dataset = Subset(ds, train_idx)
        train_dataset.data_shape = self.shape
        return train_dataset

    def validation_dataset(self) -> Dataset:
        ds = MNIST(root=MNIST_ROOT, train=True, transform=self.transform)
        val_idx = np.load(r"D:\Python\Cuboidal_Partitioning_Deep_Learning_Project\data\split_indices\MNIST_Validation_Idx.npy")
        validation_datset = Subset(ds, val_idx)
        validation_datset.data_shape = self.shape
        return validation_datset

    def test_dataset(self) -> Dataset:
        test_dataset = MNIST(root=MNIST_ROOT, train=False, transform=self.transform)
        test_dataset.data_shape = self.shape
        return test_dataset


class MyCIFAR_10(SourceDataset):
    """
    Wrapper of the CIFAR-10 dataset.
    """
    @staticmethod
    def name() -> str:
        return "CIFAR10"
    
    def __init__(self, transform: Callable | None=None) -> None:
        """
        Sets up the training, validation, and testing dataset of CIFAR-10.
        """
        self.transform = transform
        self.shape = (32, 32, 3)

    def train_dataset(self) -> Dataset:
        dataset = CIFAR10(root=CIFAR10_ROOT, train=True, transform=self.transform)
        train_idx = np.load(r"D:\Python\Cuboidal_Partitioning_Deep_Learning_Project\data\split_indices\CIFAR_Train_Idx.npy")
        train_dataset = Subset(dataset, train_idx)
        train_dataset.data_shape = self.shape
        return train_dataset

    def validation_dataset(self) -> Dataset:
        dataset = CIFAR10(root=CIFAR10_ROOT, train=True, transform=self.transform)
        val_idx = np.load(r"D:\Python\Cuboidal_Partitioning_Deep_Learning_Project\data\split_indices\CIFAR_Validation_Idx.npy")
        validation_datset = Subset(dataset, val_idx)
        validation_datset.data_shape = self.shape
        return validation_datset

    def test_dataset(self) -> Dataset:
        test_dataset = CIFAR10(root=CIFAR10_ROOT, train=False, transform=self.transform)
        test_dataset.data_shape = self.shape
        return test_dataset


class MyMedMNIST(SourceDataset):
    """
    Wrapper of the MedMNIST dataset.
    """
    @staticmethod
    def name() -> str:
        return "MedMNIST"
    
    def __init__(self, transform: Callable | None=None, size: int=28) -> None:
        """
        Sets up the training, validation, and testing dataset of MedMNIST OrganAMNIST.
        """
        self.transform = transform
        self.size = size
        self.shape = (size, size, 1)

    def train_dataset(self) -> Dataset:
        train_dataset = OrganAMNIST(root=MEDMNIST_ROOT, split="train", transform=self.transform, size=self.size)
        train_dataset.data_shape = self.shape
        return train_dataset

    def validation_dataset(self) -> Dataset:
        validation_datset = OrganAMNIST(root=MEDMNIST_ROOT, split="val", transform=self.transform, size=self.size)
        validation_datset.data_shape = self.shape
        return validation_datset

    def test_dataset(self) -> Dataset:
        test_dataset = OrganAMNIST(root=MEDMNIST_ROOT, split="test", transform=self.transform, size=self.size)
        test_dataset.data_shape = self.shape
        return test_dataset


class OmniglotDatasetFolder(DatasetFolder):
    """
    Custom dataset folder class for omniglot split recognition. This class will allocate a specific number
    of each character of each alphabet to be in the training, validation and testing dataset, maintaining
    a balance between classes in terms of characters.
    """
    def __init__(self, root: str, split: str, transform=None) -> None:
        """
        Initialises the folder structure using DatasetFolder to find classes, but saves split as an attribute to
        refer to when making the datasets
        """
        self.split = split
        super().__init__(root, default_loader, None, transform, None, None)
    
    def make_dataset(self, directory: str, class_to_idx: Dict[str, int], extensions=None, is_valid_file=None, allow_empty: bool=False):
        """
        Makes dataset for a given split by retrieving samples within a specified amount. There are 20 images per character per alphabet. 
        14 of the images will go to training, 3 will go to validation and the last 3 will go to testing.
        """
        # Create space
        instances = []

        # Go over every class
        for class_name in class_to_idx.keys():
            current_dir = directory + "/" + class_name
            
            # Go over every character in every class
            for character in os.listdir(current_dir):
                char_dir = current_dir + "/" + character

                # For each image of each character in each alphabet, append based on split
                for i, filename in enumerate(os.listdir(char_dir)):
                    if self.split == "train" and i < 14:
                        instances.append((char_dir + "/" + filename, class_to_idx[class_name]))
                    elif self.split == "val" and i > 13 and i < 17:
                        instances.append((char_dir + "/" + filename, class_to_idx[class_name]))
                    elif self.split == "test" and i > 16:
                        instances.append((char_dir + "/" + filename, class_to_idx[class_name]))
        
        return instances


class MyOmniglot(SourceDataset):
    """
    Wrapper of the Omniglot dataset.
    """
    @staticmethod
    def name() -> str:
        return "Omniglot"
    
    def __init__(self, transform: Callable | None=None) -> None:
        """
        Sets up the training, validation, and testing dataset of Omniglot.
        """
        # Turn images to Grayscale since they load as RGB
        if transform is None:
            self.transform = transforms.Grayscale()
        else:
            self.transform = transforms.Compose([transforms.Grayscale(), transform])
        self.shape = (105, 105, 1)

    def train_dataset(self) -> Dataset:
        train_dataset = OmniglotDatasetFolder(root=OMNIGLOT_ROOT, split="train", transform=self.transform)
        train_dataset.data_shape = self.shape
        return train_dataset

    def validation_dataset(self) -> Dataset:
        validation_datset = OmniglotDatasetFolder(root=OMNIGLOT_ROOT, split="val", transform=self.transform)
        validation_datset.data_shape = self.shape
        return validation_datset

    def test_dataset(self) -> Dataset:
        test_dataset = OmniglotDatasetFolder(root=OMNIGLOT_ROOT, split="test", transform=self.transform)
        test_dataset.data_shape = self.shape
        return test_dataset