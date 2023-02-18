from dataModules.dataModule import DataModule
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import torch


class MNIST_OR_16_Block_DataModule(DataModule):
    """
    A data module for the normal MNIST dataset
    """
    def __init__(self, train_dir: str, val_dir: str, batch_size: int, normalize: bool = False):
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
        self.normalize = normalize
        super().__init__(train_dir, val_dir, batch_size, DataLoader)
        

    def setup(self):
        """
        Instantiate datasets for training and validation.
        """
        array = [
            transforms.ToTensor(),
            Resize()
        ]

        if self.normalize:
            array.append(transforms.Normalize((0.1307,), (0.3081,)))

        transform = transforms.Compose(array)

        self.train_set = MNIST(root=self.train_dir, train=True, transform=transform)

        self.val_set = MNIST(root=self.val_dir, train=False, transform=transform)


class Resize():
    """
    Resize an 28 x 28 image into 4 x 4 by taking the average of each 7 x 7 section
    sections of the image into 1 pixel.
    """
    def __init__(self) -> None:
        pass

    def __call__(self, img: torch.Tensor):
        # Create space for 4 x 4 tensor
        new_img = torch.zeros(1, 4, 4)

        # Iterate over number of sections
        for i in range(4):
            for j in range(4):
                section = img[0, 7*i:(7*i+7), 7*j:(7*j+7)]
                avg = section.mean().item()
                new_img[0][i][j] = avg
        
        return new_img


#   0:7, 0:7        i=0 j=0
#   7:14, 0:7       i=0 j=1
#   14:21, 0:7      i=0 j=2
#   21:28, 0:7      i=0 j=3
#   0:7, 7:14       i=1 j=0

#   7j:7j+7, 7i:7i+7


# for u in range(28):
#     for v in range(28):
#         print(f"{int(img[0, u, v].item()*255):3}", end=', ')
#     print()