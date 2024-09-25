from torch.utils.data import Subset, Dataset
from torchvision.datasets import MNIST, CIFAR10, DatasetFolder, KMNIST
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from medmnist import OrganAMNIST
import os
import numpy as np
from typing import Callable, Dict
from PIL import Image
from skimage import transform


# Root directories to dataset sources
MNIST_ROOT = "data/source/MNIST/"
CIFAR10_ROOT = "data/source/CIFAR10/"
MEDMNIST_ROOT = "data/source/MedMNIST/"
OMNIGLOT_ROOT = "data/source/Omniglot/"
KMNIST_ROOT = "data/source/KMNIST/"
HELEN_ROOT = "data/source/HELEN/"


class SourceDataset():
    """
    Base Dataset class to encapsulate links to training, validation and test datasets.
    """
    def __init__(self, name :str, shape: tuple[int, int, int], num_classes: int) -> None:
        self.name = name
        self.shape = shape
        self.num_classes = num_classes

    def idx_to_class(self, i: int) -> str:
        pass

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

    def idx_to_class(self, i: int) -> str:
        return str(i)
        
    def train_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        ds = MNIST(root=MNIST_ROOT, train=True, transform=transform)
        train_idx = np.load("data/split_indices/MNIST_Train_Idx.npy")
        train_dataset = Subset(ds, train_idx)
        return train_dataset

    def validation_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        ds = MNIST(root=MNIST_ROOT, train=True, transform=transform)
        val_idx = np.load("data/split_indices/MNIST_Validation_Idx.npy")
        validation_datset = Subset(ds, val_idx)
        return validation_datset

    def test_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        test_dataset = MNIST(root=MNIST_ROOT, train=False, transform=transform)
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

    def idx_to_class(self, i: int) -> str:
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        return classes[i]

    def train_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        dataset = CIFAR10(root=CIFAR10_ROOT, train=True, transform=transform)
        train_idx = np.load("data/split_indices/CIFAR_Train_Idx.npy")
        train_dataset = Subset(dataset, train_idx)
        return train_dataset

    def validation_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        dataset = CIFAR10(root=CIFAR10_ROOT, train=True, transform=transform)
        val_idx = np.load("data/split_indices/CIFAR_Validation_Idx.npy")
        validation_datset = Subset(dataset, val_idx)
        return validation_datset

    def test_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        test_dataset = CIFAR10(root=CIFAR10_ROOT, train=False, transform=transform)
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

    def idx_to_class(self, i: int) -> str:
        i = i.item()
        classes = {
            '0': 'bladder',
            '1': 'femur-left',
            '2': 'femur-right',
            '3': 'heart',
            '4': 'kidney-left',
            '5': 'kidney-right',
            '6': 'liver',
            '7': 'lung-left',
            '8': 'lung-right',
            '9': 'pancreas',
            '10': 'spleen'
        }
        return classes[f"{i}"]

    def train_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        train_dataset = OrganAMNIST(root=MEDMNIST_ROOT, split="train", transform=transform, size=self.size)
        return train_dataset

    def validation_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        validation_datset = OrganAMNIST(root=MEDMNIST_ROOT, split="val", transform=transform, size=self.size)
        return validation_datset

    def test_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        test_dataset = OrganAMNIST(root=MEDMNIST_ROOT, split="test", transform=transform, size=self.size)
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

    def idx_to_class(self, i: int) -> str:
        classes = ['Bengali', 'Futurama', 'Hebrew', 'Latin']
        return classes[i]

    def train_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        train_dataset = OmniglotDatasetFolder(root=OMNIGLOT_ROOT, split="train", transform=transform)
        return train_dataset

    def validation_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        validation_datset = OmniglotDatasetFolder(root=OMNIGLOT_ROOT, split="val", transform=transform)
        return validation_datset

    def test_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        test_dataset = OmniglotDatasetFolder(root=OMNIGLOT_ROOT, split="test", transform=transform)
        return test_dataset


class MyKMNIST(SourceDataset):
    """
    Wrapper of the KMNIST dataset.
    """
    def __init__(self, transform: Callable | None=None) -> None:
        """
        Sets up the training, validation, and testing dataset of MNIST.
        """
        super().__init__(name="KMNIST", shape=(28, 28, 1), num_classes=10)
        self.transform = transform

    def idx_to_class(self, i: int) -> str:
        classes = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']
        return classes[i]
        
    def train_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        ds = KMNIST(root=KMNIST_ROOT, train=True, transform=transform)
        train_idx = np.load("data/split_indices/KMNIST_Train_Idx.npy")
        train_dataset = Subset(ds, train_idx)
        return train_dataset

    def validation_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        ds = KMNIST(root=KMNIST_ROOT, train=True, transform=transform)
        val_idx = np.load("data/split_indices/KMNIST_Validation_Idx.npy")
        validation_datset = Subset(ds, val_idx)
        return validation_datset

    def test_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        test_dataset = KMNIST(root=KMNIST_ROOT, train=False, transform=transform)
        return test_dataset


class HELEN(Dataset):
    def __init__(self, root: str, train: bool, transform=None) -> None:
        super().__init__()
        self.root = root + "train/" if train else root + "test/"
        self.transform = transform
        data = os.listdir(self.root)
        self.image_filenames = list(filter(lambda filename: filename[-4:] == ".jpg", data))
        self.landmark_filenames = list(filter(lambda filename: filename[-4:] == ".pts", data))
        self.rescale = Rescale((640, 640))

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int):
        # Get the filepaths
        image_filepath = self.root + self.image_filenames[idx]
        landmark_filepath = self.root + self.landmark_filenames[idx]

        # Load the image
        img = Image.open(image_filepath)

        # Read the landmark data
        landmarks = self.read_landmark_file(landmark_filepath)

        # Rescale image and landmarks
        img, landmarks = self.rescale(img, landmarks)

        # Apply any transforms
        if self.transform is not None:
            img = self.transform(img)

        return img, landmarks


    def read_landmark_file(self, filepath):
        with open(filepath) as f:
            # Read the .pts file
            contents = f.readlines()

            # Only get the numbers
            contents = contents[3:71]

            # Put all the coordinates together
            contents = "".join(contents).replace("\n", " ")
            contents = contents[:-1]
            landmark_data = [float(val) for val in contents.split()]

            return landmark_data


class MyHELEN(SourceDataset):
    def __init__(self, transform: Callable | None=None) -> None:
        """
        Sets up the training, validation, and testing dataset of Omniglot.
        """
        super().__init__(name="HELEN", shape=(640, 640, 3), num_classes=68*2)
        self.transform = transform

    def train_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        dataset = HELEN(root=HELEN_ROOT, train=True, transform=transform)
        train_idx = np.load("data/split_indices/HELEN_Train_Idx.npy")
        train_dataset = Subset(dataset, train_idx)
        return train_dataset

    def validation_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        dataset = HELEN(root=HELEN_ROOT, train=True, transform=transform)
        validation_idx = np.load("data/split_indices/HELEN_Validation_Idx.npy")
        validation_dataset = Subset(dataset, validation_idx)
        return validation_dataset

    def test_dataset(self, transform=None) -> Dataset:
        if transform is None:
            transform = self.transform
        test_dataset = HELEN(root=HELEN_ROOT, train=False, transform=transform)
        return test_dataset


class Rescale():
    """
    Rescale the image in a sample to a given size.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image, landmarks):
        # Transform img and landmarks into usable numpy array
        image = np.asarray(image)
        landmarks = np.array(landmarks, dtype=np.float64).reshape((-1, 2))

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        img = transform.resize(image, (new_h, new_w), anti_aliasing=new_h < h and new_w < w, clip=True, preserve_range=True).astype(np.uint8)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * np.array([new_w / w, new_h / h], dtype=np.float64)

        return img, landmarks.reshape((-1)).tolist()
