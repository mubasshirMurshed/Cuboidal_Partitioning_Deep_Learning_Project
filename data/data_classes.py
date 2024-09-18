from torch.utils.data import Subset, Dataset
from torchvision.datasets import MNIST, CIFAR10, DatasetFolder
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from medmnist import OrganAMNIST
import os
import numpy as np
from typing import Callable, Dict
from abc import ABC, abstractmethod


# Root directories to dataset sources
MNIST_ROOT = "data/source/MNIST/"
CIFAR10_ROOT = "data/source/CIFAR10/"
MEDMNIST_ROOT = "data/source/MedMNIST/"
OMNIGLOT_ROOT = "data/source/Omniglot/"


class SourceDataset():
    """
    Base Dataset class to encapsulate links to training, validation and test datasets.
    """
    def __init__(self, name :str, shape: tuple[int, int, int], num_classes: int) -> None:
        self.name = name
        self.shape = shape
        self.num_classes = num_classes

    def train_dataset(self, transform=None) -> Dataset:
        pass

    def validation_dataset(self, transform=None) -> Dataset:
        pass

    def test_dataset(self, transform=None) -> Dataset:
        pass
        

class MyMNIST(SourceDataset):
    """
    Wrapper of the MNIST dataset.
    """
    def __init__(self, transform: Callable | None=None) -> None:
        """
        Sets up the training, validation, and testing dataset of MNIST.
        """
        super().__init__(name="MNIST", shape=(28, 28, 1), num_classes=10)
        self.transform = transform
        
    def train_dataset(self, transform=None) -> Dataset:
        if self.transform is None:
            self.transform = transform
        ds = MNIST(root=MNIST_ROOT, train=True, transform=self.transform)
        train_idx = np.load("data/split_indices/MNIST_Train_Idx.npy")
        train_dataset = Subset(ds, train_idx)
        train_dataset.data_shape = self.shape
        return train_dataset

    def validation_dataset(self, transform=None) -> Dataset:
        if self.transform is None:
            self.transform = transform
        ds = MNIST(root=MNIST_ROOT, train=True, transform=self.transform)
        val_idx = np.load("data/split_indices/MNIST_Validation_Idx.npy")
        validation_datset = Subset(ds, val_idx)
        validation_datset.data_shape = self.shape
        return validation_datset

    def test_dataset(self, transform=None) -> Dataset:
        if self.transform is None:
            self.transform = transform
        test_dataset = MNIST(root=MNIST_ROOT, train=False, transform=self.transform)
        test_dataset.data_shape = self.shape
        return test_dataset


class MyCIFAR_10(SourceDataset):
    """
    Wrapper of the CIFAR-10 dataset.
    """   
    def __init__(self, transform: Callable | None=None) -> None:
        """
        Sets up the training, validation, and testing dataset of CIFAR-10.
        """
        super().__init__(name="CIFAR10", shape=(32, 32, 3), num_classes=10)
        self.transform = transform

    def train_dataset(self, transform=None) -> Dataset:
        if self.transform is None:
            self.transform = transform
        dataset = CIFAR10(root=CIFAR10_ROOT, train=True, transform=self.transform)
        train_idx = np.load("data/split_indices/CIFAR_Train_Idx.npy")
        train_dataset = Subset(dataset, train_idx)
        train_dataset.data_shape = self.shape
        return train_dataset

    def validation_dataset(self, transform=None) -> Dataset:
        if self.transform is None:
            self.transform = transform
        dataset = CIFAR10(root=CIFAR10_ROOT, train=True, transform=self.transform)
        val_idx = np.load("data/split_indices/CIFAR_Validation_Idx.npy")
        validation_datset = Subset(dataset, val_idx)
        validation_datset.data_shape = self.shape
        return validation_datset

    def test_dataset(self, transform=None) -> Dataset:
        if self.transform is None:
            self.transform = transform
        test_dataset = CIFAR10(root=CIFAR10_ROOT, train=False, transform=self.transform)
        test_dataset.data_shape = self.shape
        return test_dataset


class MyMedMNIST(SourceDataset):
    """
    Wrapper of the MedMNIST dataset.
    """   
    def __init__(self, transform: Callable | None=None, size: int=28) -> None:
        """
        Sets up the training, validation, and testing dataset of MedMNIST OrganAMNIST.
        """
        super().__init__(name="MedMNIST", shape=(size, size, 1), num_classes=11)
        self.transform = transform
        self.size = size

    def train_dataset(self, transform=None) -> Dataset:
        if self.transform is None:
            self.transform = transform
        train_dataset = OrganAMNIST(root=MEDMNIST_ROOT, split="train", transform=self.transform, size=self.size)
        train_dataset.data_shape = self.shape
        return train_dataset

    def validation_dataset(self, transform=None) -> Dataset:
        if self.transform is None:
            self.transform = transform
        validation_datset = OrganAMNIST(root=MEDMNIST_ROOT, split="val", transform=self.transform, size=self.size)
        validation_datset.data_shape = self.shape
        return validation_datset

    def test_dataset(self, transform=None) -> Dataset:
        if self.transform is None:
            self.transform = transform
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
    def __init__(self, transform: Callable | None=None) -> None:
        """
        Sets up the training, validation, and testing dataset of Omniglot.
        """
        super().__init__(name="Omniglot", shape=(105, 105, 1), num_classes=4)
        
        # Turn images to Grayscale since they load as RGB
        if transform is None:
            self.transform = transforms.Grayscale()
        else:
            self.transform = transforms.Compose([transforms.Grayscale(), transform])

    def train_dataset(self, transform=None) -> Dataset:
        if self.transform is None:
            self.transform = transform
        train_dataset = OmniglotDatasetFolder(root=OMNIGLOT_ROOT, split="train", transform=self.transform)
        train_dataset.data_shape = self.shape
        return train_dataset

    def validation_dataset(self, transform=None) -> Dataset:
        if self.transform is None:
            self.transform = transform
        validation_datset = OmniglotDatasetFolder(root=OMNIGLOT_ROOT, split="val", transform=self.transform)
        validation_datset.data_shape = self.shape
        return validation_datset

    def test_dataset(self, transform=None) -> Dataset:
        if self.transform is None:
            self.transform = transform
        test_dataset = OmniglotDatasetFolder(root=OMNIGLOT_ROOT, split="test", transform=self.transform)
        test_dataset.data_shape = self.shape
        return test_dataset
