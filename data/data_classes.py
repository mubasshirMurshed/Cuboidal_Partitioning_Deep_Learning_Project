import torch
from torch.utils.data import random_split
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from medmnist import OrganAMNIST

MNIST_ROOT = "data/source/MNIST"
CIFAR_10_ROOT = "data/source/CIFAR-10/"
MEDMNIST_ROOT = "data/source/MedMNIST/"
OMNIGLOT_ROOT = "data/source/Omniglot/"

generator = torch.Generator().manual_seed(42)

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
        self.train_ds, self.validation_ds = random_split(ds, [50000, 10000], generator=generator)
        self.test_ds = MNIST(root=MNIST_ROOT, train=False, transform=transform)
        self.shape = (28, 28, 1)
        super().__init__()


class MyCIFAR_10(Dataset):
    def __init__(self, transform=None) -> None:
        # TODO: Add fixed deterministic logic through file reading
        ds = CIFAR10(root=CIFAR_10_ROOT, train=True, transform=transform)
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


class MyOmniglot(Dataset):
    def __init__(self, transform=None) -> None:
        # TODO: Add fixed deterministic logic through file reading
        ds = ImageFolder(root=OMNIGLOT_ROOT, transform=transform)
        self.train_ds, self.validation_ds, self.test = random_split(ds, [0.7, 0.2, 0.1], generator=generator)
        self.shape = (105, 105, 1)
        super().__init__()