import torch
from torch.utils.data import random_split
from torchvision.datasets import MNIST, CIFAR10, DatasetFolder
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from medmnist import OrganAMNIST
import os

MNIST_ROOT = "data/source/MNIST"
CIFAR_10_ROOT = "data/source/CIFAR-10/"
MEDMNIST_ROOT = "data/source/MedMNIST/"
OMNIGLOT_ROOT = "data/source/Omniglot/"

class Dataset:
    def __init__(self):
        self.train_ds.shape = self.shape
        self.validation_ds.shape = self.shape
        self.test_ds.shape = self.shape

    def train(self):
        return self.train_ds

    def validation(self):
        return self.validation_ds

    def test(self):
        return self.test_ds


class MyMNIST(Dataset):
    def __init__(self, transform=None) -> None:
        # TODO: Add fixed deterministic logic through file reading
        ds = MNIST(root=MNIST_ROOT, train=True, transform=transform)
        generator = torch.Generator().manual_seed(42)
        self.train_ds, self.validation_ds = random_split(ds, [50000, 10000], generator=generator)
        self.test_ds = MNIST(root=MNIST_ROOT, train=False, transform=transform)
        self.shape = (28, 28, 1)
        super().__init__()


class MyCIFAR_10(Dataset):
    def __init__(self, transform=None) -> None:
        # TODO: Add fixed deterministic logic through file reading
        ds = CIFAR10(root=CIFAR_10_ROOT, train=True, transform=transform)
        generator = torch.Generator().manual_seed(42)
        self.train_ds, self.validation_ds = random_split(ds, [50000, 10000], generator=generator)
        self.test_ds = CIFAR10(root=CIFAR_10_ROOT, train=False, transform=transform)
        self.shape = (32, 32, 3)
        super().__init__()


class MyMedMNIST(Dataset):
    def __init__(self, transform=None, size=None) -> None:
        self.train_ds = OrganAMNIST(root=MEDMNIST_ROOT, split="train", transform=transform, size=size)
        self.validation_ds = OrganAMNIST(root=MEDMNIST_ROOT, split="val", transform=transform, size=size)
        self.test_ds = OrganAMNIST(root=MEDMNIST_ROOT, split="test", transform=transform, size=size)
        self.shape = (size, size, 1)
        super().__init__()


class OmniglotDatasetFolder(DatasetFolder):
    def __init__(self, root: str, split: str, transform=None) -> None:
        self.split = split
        super().__init__(root, default_loader, None, transform, None, None)
    
    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None, allow_empty=False):
        instances = []
        for class_name in class_to_idx.keys():
            current_dir = directory + "/" + class_name
            for character in os.listdir(current_dir):
                char_dir = current_dir + "/" + character
                for i, filename in enumerate(os.listdir(char_dir)):
                    if self.split == "train" and i < 14:
                        instances.append((char_dir + "/" + filename, class_to_idx[class_name]))
                    elif self.split == "val" and i > 13 and i < 17:
                        instances.append((char_dir + "/" + filename, class_to_idx[class_name]))
                    elif self.split == "test" and i > 16:
                        instances.append((char_dir + "/" + filename, class_to_idx[class_name]))
        return instances


class MyOmniglot(Dataset):
    def __init__(self, transform=None) -> None:
        if transform is None:
            transform = transforms.Grayscale()
        else:
            transform = transforms.Compose([transforms.Grayscale(), transform])
        self.train_ds = OmniglotDatasetFolder(root=OMNIGLOT_ROOT, split="train", transform=transform)
        self.validation_ds = OmniglotDatasetFolder(root=OMNIGLOT_ROOT, split="val", transform=transform)
        self.test_ds = OmniglotDatasetFolder(root=OMNIGLOT_ROOT, split="test", transform=transform)
        self.shape = (105, 105, 1)
        super().__init__()