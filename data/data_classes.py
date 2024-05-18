from torch.utils.data import Subset, Dataset
from torchvision.datasets import MNIST, CIFAR10, DatasetFolder
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from medmnist import OrganAMNIST
import os
import numpy as np
from typing import Callable, Dict

# Root directories to dataset sources       # TODO: Fix these to not be abs paths
MNIST_ROOT = r"D:\Python\Cuboidal_Partitioning_Deep_Learning_Project\data\source\MNIST"
CIFAR_10_ROOT = r"D:\Python\Cuboidal_Partitioning_Deep_Learning_Project\data\source\CIFAR-10"
MEDMNIST_ROOT = r"D:\Python\Cuboidal_Partitioning_Deep_Learning_Project\data\source\MedMNIST"
OMNIGLOT_ROOT = r"D:\Python\Cuboidal_Partitioning_Deep_Learning_Project\data\source\Omniglot"

"""
Base Dataset class to encapsulate links to training, validation and test datasets.
"""
class Dataset:
    def __init__(self) -> None:
        """
        Provide shapes to each of the dataset splits
        """
        self.train_ds.shape = self.shape
        self.validation_ds.shape = self.shape
        self.test_ds.shape = self.shape

    def train(self) -> Dataset:
        """
        Retrieve training dataset
        """
        return self.train_ds

    def validation(self) -> Dataset:
        """
        Retrieve validation dataset
        """
        return self.validation_ds

    def test(self) -> Dataset:
        """
        Retrieve testing dataset
        """
        return self.test_ds


"""
Wrapper of the MNIST dataset.
"""
class MyMNIST(Dataset):
    def __init__(self, transform: Callable | None=None) -> None:
        """
        Sets up the training, validation, and testing dataset of MNIST.
        """
        # Get dataset source
        ds = MNIST(root=MNIST_ROOT, train=True, transform=transform)

        # Load in pre-defined subset indices for train/val split
        train_idx = np.load(r"D:\Python\Cuboidal_Partitioning_Deep_Learning_Project\data\split_indices\MNIST_Train_Idx.npy")
        val_idx = np.load(r"D:\Python\Cuboidal_Partitioning_Deep_Learning_Project\data\split_indices\MNIST_Validation_Idx.npy")

        # Obtain each dataset split
        self.train_ds = Subset(ds, train_idx)
        self.validation_ds = Subset(ds, val_idx)
        self.test_ds = MNIST(root=MNIST_ROOT, train=False, transform=transform)
        self.shape = (28, 28, 1)
        super().__init__()


"""
Wrapper of the CIFAR-10 dataset.
"""
class MyCIFAR_10(Dataset):
    def __init__(self, transform: Callable | None=None) -> None:
        """
        Sets up the training, validation, and testing dataset of CIFAR-10.
        """
        # Get dataset source
        ds = CIFAR10(root=CIFAR_10_ROOT, train=True, transform=transform)

        # Load in pre-defined subset indices for train/val split
        train_idx = np.load(r"D:\Python\Cuboidal_Partitioning_Deep_Learning_Project\data\split_indices\CIFAR_Train_Idx.npy")
        val_idx = np.load(r"D:\Python\Cuboidal_Partitioning_Deep_Learning_Project\data\split_indices\CIFAR_Validation_Idx.npy")

        # Obtain each dataset split
        self.train_ds = Subset(ds, train_idx)
        self.validation_ds = Subset(ds, val_idx)
        self.test_ds = CIFAR10(root=CIFAR_10_ROOT, train=False, transform=transform)
        self.shape = (32, 32, 3)
        super().__init__()


"""
Wrapper of the MedMNIST dataset.
"""
class MyMedMNIST(Dataset):
    def __init__(self, transform: Callable | None=None, size: int | None=None) -> None:
        """
        Sets up the training, validation, and testing dataset of MedMNIST OrganAMNIST.
        """
        # Obtain each dataset split
        self.train_ds = OrganAMNIST(root=MEDMNIST_ROOT, split="train", transform=transform, size=size)
        self.validation_ds = OrganAMNIST(root=MEDMNIST_ROOT, split="val", transform=transform, size=size)
        self.test_ds = OrganAMNIST(root=MEDMNIST_ROOT, split="test", transform=transform, size=size)
        self.shape = (size, size, 1)
        super().__init__()


"""
Custom dataset folder class for omniglot split recognition. This class will allocate a specific number
of each character of each alphabet to be in the training, validation and testing dataset, maintaining
a balance between classes in terms of characters.
"""
class OmniglotDatasetFolder(DatasetFolder):
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


"""
Wrapper of the Omniglot dataset.
"""
class MyOmniglot(Dataset):
    def __init__(self, transform: Callable | None=None) -> None:
        """
        Sets up the training, validation, and testing dataset of Omniglot.
        """
        # Turn images to Grayscale since they load as RGB
        if transform is None:
            transform = transforms.Grayscale()
        else:
            transform = transforms.Compose([transforms.Grayscale(), transform])

        # Obtain each dataset split through custom class
        self.train_ds = OmniglotDatasetFolder(root=OMNIGLOT_ROOT, split="train", transform=transform)
        self.validation_ds = OmniglotDatasetFolder(root=OMNIGLOT_ROOT, split="val", transform=transform)
        self.test_ds = OmniglotDatasetFolder(root=OMNIGLOT_ROOT, split="test", transform=transform)
        self.shape = (105, 105, 1)
        super().__init__()