# Imports
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

class MNISTCuboidalSparseDataset(Dataset):
    """
    A class that loads in the partitioned data of MNIST images all into RAM where each image
    is partitioned into n cuboids, each of which has 5 features of data attached.
    
    The data is downloaded by row from the cvs file and rearranges the shape to 128 x 5 tensor
    and passed this matrix and the corresponding label.
    """
    def __init__(self, csv_file_dir: str, n: int, length: int) -> None:
        """
        Loads all image data into memory at once by reading a given csv file of partitioning data.
        
        Args:
        - csv_file_dir: str
            - The file path directory to the csv file containing the data
        - n: int
            - The number of cuboids the images have been partitioned into
        - length: int
            - A given ceiling on the number of images to uses from the csv file
        """
        # Give UI information
        print("Loading in dataset in memory...")

        # Read in data file and save attributes
        self.data = pd.read_csv(csv_file_dir, header=None, nrows=length)
        self.cuboids = n

        # Fill in self.dataset with the appropriate reshapes of the data
        self.dataset = [0]*len(self.data)
        for index in tqdm(range(len(self.data)), leave=False):
            # Generate space for data in from of 128 x 5 grid
            matrix = [0]*(self.cuboids)

            # Find centre and add data to vector
            for i in range(self.cuboids):
                vector = [0]*5
                for j in range(5):
                    dataItem = self.data.values[index][i + j*self.cuboids]

                    # Define normalizer
                    if (j < 4):
                        std_divider = 28
                    else:
                        std_divider = 255

                    dataItem = dataItem/std_divider
                    vector[j] = dataItem
                matrix[i] = vector
            
            # Find label
            label = self.data.values[index][self.cuboids*5]

            # Convert vector to tensor
            image = torch.tensor(matrix)
            image = image.float()

            # Stored transformed data and label in a tuple
            self.dataset[index] = (image, label)
        
        # Print separator lines
        print('-' * 20)

    def __len__(self):
        """
        Return length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Retrieves partitioning data of a selected image and its label by accessing internally
        stored data.

        Args:
        - index: int
            - Index of image
        """
        return self.dataset[index]