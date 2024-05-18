# Imports
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from typing import Optional
from .transforms import CuPIDPartition
from torchvision.datasets import MNIST
from torch.utils.data import random_split
import math
import multiprocessing as mp

# TODO: Fourth form that is super on the fly, not in memory

class MNISTGraphDataset_Auto(InMemoryDataset):
    """
    In memory dataset class that stores the dataset in a .pt file under
    root/processed.

    The class will ignore information if it detects that the .pt file is already
    created by specifying its name in the processed_file_names attribute.

    Selection of which feature data to include can be done via flags. By default, every value is
    included.
    """
    def __init__(self, root: Optional[str]=None,
                       num_cuboids: Optional[int]=None,
                       mode: Optional[str]=None, 
                       split: Optional[str] = None,
                       train_length: int=50000,
                       colour: bool=False,
                       x_centre: bool=False,
                       y_centre: bool=False,
                       num_pixels: bool=False,
                       angle: bool=False,
                       width: bool=False,
                       height: bool=False) -> None:
        """
        Saves attributes and runs super init to do processing and loading of the data in
        self.data.

        Args:
        - root: str | None
            - The string path to the root folder where the data files are
        - name: str | None
            - The group name of the file to find
        - mode: str | None
            - The type of dataset being loaded, CP (cuboidal partition) or SP
              (superpixel partition). Other modes also exist such as RP (regular
              partition).
        - partition_limit: int | None
            - The upper limit of number of segments allowed for each image
        - colour: bool
            - Whether to include colour data in each graph node
        - x_centre: bool
            - Whether to include x-centre data in each graph node
        - y_centre: bool
            - Whether to include y centre data in each graph node
        - num_pixels: bool
            - Whether to include number of pixels of a cuboid in its corresponding graph node
        - angle: bool
            - Whether to include the angle of the cuboid in its corresponding graph node
        - width: bool
            - Whether to include the width of the cuboid in its corresponding graph node
        - height: bool
            - Whether to include the height of the cuboid in its corresponding graph node
        """
        # Save attributes
        self.num_cuboids = num_cuboids
        self.root = root + "mnist" + str(self.num_cuboids) + "/"         # The dataset processed files will be saved here
        self.source_root = root + "mnistPytorch/"                   # The path to MNIST dataset provided by PyTorch
        self.mode = mode
        self.split = split
        self.train_length = train_length
        self.val_length = 60000 - self.train_length
        self.test_length = 10000
        
        # Ablation
        self.colour = colour
        self.x_centre = x_centre
        self.y_centre = y_centre
        self.num_pixels = num_pixels
        self.angle = angle
        self.width = width
        self.height = height

        # Create ablation code string
        self.ablation_code = ""
        if self.x_centre:
            self.ablation_code += 'X'
        if self.y_centre:
            self.ablation_code += 'Y'
        if self.colour:
            self.ablation_code += 'C'
        if self.num_pixels:
            self.ablation_code += 'N'
        if self.angle:
            self.ablation_code += 'A'
        if self.width:
            self.ablation_code += 'W'
        if self.height:
            self.ablation_code += 'H'

        super().__init__(root)

        # Load dataset as self.data
        self.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        """
        The name of the file which has the processed and saved data.
        """
        return [f'mnist{self.split}-{self.mode}-{self.num_cuboids}-{self.train_length}-{self.ablation_code}.pt']


    def process(self):
        """
        Reads in the data from the given filename and loads it all in an array of Data objects
        which is then collated to save easy in a .pt file.

        This features data extraction from the csv file,, normalization of the values, and the creation
        of the Data object for each image, including its feature matrix, adjacency matrix and label.
        """
        # Give UI information
        print("Loading in dataset in memory...")

        # MNIST Data Source with CuPID
        self.dataset = MNIST(root=self.source_root, train=self.split in ["Train", "Validation"], transform=CuPIDPartition(self.num_cuboids), download=True)

        # Deal with train and validation split
        if self.split in ["Train", "Validation"]:
            generator = torch.Generator().manual_seed(42)
            self.train, self.val = random_split(self.dataset, [self.train_length, self.val_length], generator=generator)
            if self.split == "Train":
                self.dataset = self.train
            else:
                self.dataset = self.val

        # Make space for data objects from this file
        dataset = [0]*(len(self.dataset))

        # Convert each CuPID image into a graph
        for i in tqdm(range(len(self.dataset)), leave=False):
            # Current CuPID data
            cupid_table, label = self.dataset[i]

            # Filter table to only contain leaf cuboids
            cupid_table = cupid_table[cupid_table["order"] < 0].sort_values(by="order", ascending=False)
            no_of_nodes = len(cupid_table)

            # Fill in feature matrix
            x = torch.zeros((no_of_nodes, len(self.ablation_code)))
            for i in range(len(cupid_table)):
                row = cupid_table.iloc[i]
                for j in range(len(self.ablation_code)):
                    if self.ablation_code[j] == 'X':
                        x[i][j] = (row["size"][1]+1)/2 + row["start"][1] + 1
                        x[i][j] /= 28
                    elif self.ablation_code[j] == 'Y':
                        x[i][j] = (row["size"][0]+1)/2 + row["start"][0] + 1
                        x[i][j] /= 28
                    elif self.ablation_code[j] == 'C':
                        x[i][j] =  round(row["mu"].item())
                        x[i][j] /= 255
                    elif self.ablation_code[j] == 'N':
                        x[i][j] =  row["n"]
                        x[i][j] /= (28*28)
                    elif self.ablation_code[j] == 'A':
                        x[i][j] = round(math.atan(row["size"][0]/row["size"][1])*180/math.pi, 2)
                        x[i][j] /= 90
                    elif self.ablation_code[j] == 'W':
                        x[i][j] = row["size"][1]
                        x[i][j] /= 28
                    elif self.ablation_code[j] == 'H':
                        x[i][j] = row["size"][0]
                        x[i][j] /= 28
            x = x.float()

            # Find edges in COO format
            coo_src = []
            coo_dst = []
            for i in range(len(cupid_table)):
                for j in range(len(cupid_table)):
                    if i == j:
                        continue
                    row1 = cupid_table.iloc[i]
                    row2 = cupid_table.iloc[j]

                    # Construct cuboid's x_centre, y_centre, width and height
                    c1 = ((row1["size"][1]+1)/2 + row1["start"][1] + 1,
                        (row1["size"][0]+1)/2 + row1["start"][0] + 1,
                        row1["size"][1],
                        row1["size"][0]
                        )

                    c2 = ((row2["size"][1]+1)/2 + row2["start"][1] + 1,
                        (row2["size"][0]+1)/2 + row2["start"][0] + 1,
                        row2["size"][1],
                        row2["size"][0]
                        )
                    
                    if adjacent(c1, c2):
                        coo_src.append(-1*row2["order"] - 1)
                        coo_dst.append(-1*row1["order"] - 1)
            edge_index = torch.tensor([coo_src, coo_dst], dtype=torch.int64)

            # Create Data object
            label = torch.tensor(int(label))
            dataset[i] = Data(x=x, y=label, edge_index=edge_index)

        # Collate data into one massive Data object and save its state
        self.save(dataset, self.processed_paths[0])

        # Print UI Information
        print(f"Dataset loaded!")
        
        # Print separator lines
        print('-' * 20)


class MNISTGraphDataset_Auto_Parallel(InMemoryDataset):
    """
    In memory dataset class that stores the dataset in a .pt file under
    root/processed.

    The class will ignore information if it detects that the .pt file is already
    created by specifying its name in the processed_file_names attribute.

    Selection of which feature data to include can be done via flags. By default, every value is
    included.
    """
    def __init__(self, root: Optional[str]=None,
                       num_cuboids: Optional[int]=None,
                       mode: Optional[str]=None, 
                       split: Optional[str] = None,
                       train_length: int=50000,
                       colour: bool=False,
                       x_centre: bool=False,
                       y_centre: bool=False,
                       num_pixels: bool=False,
                       angle: bool=False,
                       width: bool=False,
                       height: bool=False) -> None:
        """
        Saves attributes and runs super init to do processing and loading of the data in
        self.data.

        Args:
        - root: str | None
            - The string path to the root folder where the data files are
        - name: str | None
            - The group name of the file to find
        - mode: str | None
            - The type of dataset being loaded, CP (cuboidal partition) or SP
              (superpixel partition). Other modes also exist such as RP (regular
              partition).
        - partition_limit: int | None
            - The upper limit of number of segments allowed for each image
        - colour: bool
            - Whether to include colour data in each graph node
        - x_centre: bool
            - Whether to include x-centre data in each graph node
        - y_centre: bool
            - Whether to include y centre data in each graph node
        - num_pixels: bool
            - Whether to include number of pixels of a cuboid in its corresponding graph node
        - angle: bool
            - Whether to include the angle of the cuboid in its corresponding graph node
        - width: bool
            - Whether to include the width of the cuboid in its corresponding graph node
        - height: bool
            - Whether to include the height of the cuboid in its corresponding graph node
        """
        # Save attributes
        self.num_cuboids = num_cuboids
        self.root = root + "mnist" + str(self.num_cuboids) + "/"         # The dataset processed files will be saved here
        self.source_root = root + "mnistPytorch/"                   # The path to MNIST dataset provided by PyTorch
        self.mode = mode
        self.split = split
        self.train_length = train_length
        self.val_length = 60000 - self.train_length
        self.test_length = 10000
        
        # Ablation
        self.colour = colour
        self.x_centre = x_centre
        self.y_centre = y_centre
        self.num_pixels = num_pixels
        self.angle = angle
        self.width = width
        self.height = height

        # Create ablation code string
        self.ablation_code = ""
        if self.x_centre:
            self.ablation_code += 'X'
        if self.y_centre:
            self.ablation_code += 'Y'
        if self.colour:
            self.ablation_code += 'C'
        if self.num_pixels:
            self.ablation_code += 'N'
        if self.angle:
            self.ablation_code += 'A'
        if self.width:
            self.ablation_code += 'W'
        if self.height:
            self.ablation_code += 'H'

        super().__init__(root)

        # Load dataset as self.data
        self.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        """
        The name of the file which has the processed and saved data.
        """
        return [f'mnist{self.split}-{self.mode}-{self.num_cuboids}-{self.train_length}-{self.ablation_code}.pt']


    def process(self):
        """
        Reads in the data from the given filename and loads it all in an array of Data objects
        which is then collated to save easy in a .pt file.

        This features data extraction from the csv file,, normalization of the values, and the creation
        of the Data object for each image, including its feature matrix, adjacency matrix and label.
        """
        # Give UI information
        print("Loading in dataset in memory...")

        # MNIST Data Source with CuPID
        self.dataset = MNIST(root=self.source_root, train=self.split in ["Train", "Validation"], transform=CuPIDPartition(self.num_cuboids), download=True)

        # Deal with train and validation split
        if self.split in ["Train", "Validation"]:
            generator = torch.Generator().manual_seed(42)
            self.train, self.val = random_split(self.dataset, [self.train_length, self.val_length], generator=generator)
            if self.train_val_test == 0:
                self.dataset = self.train
            else:
                self.dataset = self.val

        # Convert each CuPID image into a graph
        pool = mp.Pool(30)
        dataset = list(tqdm(pool.imap(cupid_to_graph, zip(range(len(self.dataset)), [self.dataset]*len(self.dataset), [self.ablation_code]*len(self.dataset)), 1), total=len(self.dataset), leave=False))               

        # Collate data into one massive Data object and save its state
        self.save(dataset, self.processed_paths[0])

        # Print UI Information
        print(f"Dataset loaded!")
        
        # Print separator lines
        print('-' * 20)


"""
In memory dataset class that stores the dataset in a .pt file under
root/dataset/num_segments/processed. Assumes .csv files exist in
root/dataset/num_segments/raw.

The class will ignore information if it detects that the .pt file is already
created by specifying its name in the processed_file_names attribute.

Selection of which feature data to include can be done via flags. By default, every value is
included.
"""
class Graph_Dataset_CSV(InMemoryDataset):
    
    def __init__(self, root: str, name: str, split: str, mode: str, num_segments: int, length: int | None=None,
                       x_center: bool=False, y_center: bool=False, colour: bool=False, width: bool=False, height: bool=False,
                       num_pixels: bool=False, angle: bool=False, st_dev: bool=False) -> None:
        """
        Saves attributes and runs super init to do processing and loading of the data done by super class.

        Args:
        - root: str
            - The string path to the root folder where all the data is
        - name: str
            - The name of the dataset to look for
        - split: str
            - Train/validation/test
        - mode: str
            - The type of partitioning
        - num_segments: int
            - How many segments to partition the dataset into
        - length: int | None
            - Length of dataset to consider
        - x_center: bool
            - Whether to include x-center data in each graph node
        - y_center: bool
            - Whether to include y-center data in each graph node
        - colour: bool
            - Whether to include colour data in each graph node
        - num_pixels: bool
            - Whether to include size data in each graph node
        - angle: bool
            - Whether to include angle data in each graph node
        - width: bool
            - Whether to include width data in each graph node
        - height: bool
            - Whether to include height data in each graph node
        - st_dev: bool
            - Whether to include standard deviation data in each graph node
        """
        # Save attributes
        self.root = root + name + "/" + str(num_segments) + "/"
        self.name = name
        self.length = length if length is not None else 10000           # TODO: Fix this to be more accurate, and for setting actual short lengths
        self.mode = mode
        self.num_segments = num_segments
        self.split = split

        # Ablation attributes
        self.colour = colour
        self.x_center = x_center
        self.y_center = y_center
        self.num_pixels = num_pixels
        self.angle = angle
        self.width = width
        self.height = height
        self.stdev = st_dev

        # Create ablation code string
        self.ablation_code = ""
        if self.x_center:
            self.ablation_code += 'X'
        if self.y_center:
            self.ablation_code += 'Y'
        if self.colour:
            self.ablation_code += 'C'
        if self.num_pixels:
            self.ablation_code += 'N'
        if self.angle:
            self.ablation_code += 'A'
        if self.width:
            self.ablation_code += 'W'
        if self.height:
            self.ablation_code += 'H'
        if self.stdev:
            self.ablation_code += 'S'

        # Run inherited processes to create dataset .pt files
        super().__init__(self.root)

        # Load dataset as self.data
        self.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        """
        List of the raw file names to process and apply transforms. In this case
        our data is already given as csv files.
        """
        directory = os.listdir(f"{self.root}/raw")
        filtered = filter(lambda file: self.split in file and f"-{self.mode}-" in file, directory)
        return list(filtered)


    @property
    def processed_file_names(self):
        """
        The name of the file which has the processed and saved data.
        """
        return [f'{self.name}{self.split}-{self.mode}-{self.num_segments}-{self.length*len(self.raw_file_names)}-{self.ablation_code}.pt'] #TODO: Actual length not calculated correctly


    def process(self):
        """
        Reads in the data from the given filename and loads it all in an array of Data objects
        which is then collated in a .pt file.

        This features data extraction from the csv file, normalization of the values to 0-1, and the creation
        of the Data object for each image, including its feature matrix, adjacency matrix and label.
        """
        # Give UI information
        print(f"Loading in {self.split} Dataset in memory...")
        full_data_list = []

        # Open each filepath in a for loop to get max number of columns
        num_cols = [0]*len(self.raw_paths)
        for i, filepath in enumerate(self.raw_paths):
            # Find max number of columns in csv
            f = open(filepath)
            num_cols[i] = max(len(line.split(',')) for line in f)
            f.close()

        # Iterate through each file
        for i, filepath in enumerate(self.raw_paths):
            # Read in data file and save attributes
            if self.length is not None:
                df = pd.read_csv(filepath, nrows=self.length, header=None, skiprows=[0], names=range(num_cols[i])).fillna(0)
            else:
                df = pd.read_csv(filepath, header=None, skiprows=[0], names=range(num_cols[i])).fillna(0)

            # Make space for data objects from this file
            data_list = [0]*(len(df))

            # Create each Data object and store it
            for i in tqdm(range(len(df)), leave=False):
                # Current Image
                img_data = df.iloc[i]

                # Create space for node feature matrix
                feature_matrix = []
                num_nodes = int(img_data[3])
                num_colours = int(img_data[4])
                shape_of_data = (int(img_data[5]), int(img_data[6]))
                start = 7
                num_features = len(self.ablation_code)

                # Add more based on features that are not scalars
                if self.colour:
                    num_features += (num_colours - 1)
                if self.stdev:
                    num_features += (num_colours - 1)

                # Add values into feature matrix
                if self.x_center:
                    X = img_data[start:start+num_nodes]
                    X = X / shape_of_data[1]  # Normalization
                    feature_matrix.extend(X)
                if self.y_center:
                    Y = img_data[start+num_nodes:start+num_nodes*2]
                    Y = Y / shape_of_data[0]  # Normalization
                    feature_matrix.extend(Y)
                if self.colour:
                    for j in range(num_colours):
                        C = img_data[start+num_nodes*(2+j):start+num_nodes*(3+j)]
                        C = C / 255  # Normalization
                        feature_matrix.extend(C)
                if self.num_pixels:
                    N = img_data[start+num_nodes*(2+num_colours):start+num_nodes*(3+num_colours)]
                    N = N / np.prod(shape_of_data)  # Normalization
                    feature_matrix.extend(N)
                if self.angle:
                    A = img_data[start+num_nodes*(3+num_colours):start+num_nodes*(4+num_colours)]
                    A = A / 90  # Normalization
                    feature_matrix.extend(A)
                if self.width:
                    W = img_data[start+num_nodes*(4+num_colours):start+num_nodes*(5+num_colours)]
                    W = W / shape_of_data[1]  # Normalization
                    feature_matrix.extend(W)
                if self.height:
                    H = img_data[start+num_nodes*(5+num_colours):start+num_nodes*(6+num_colours)]
                    H = H / shape_of_data[0]  # Normalization
                    feature_matrix.extend(H)
                if self.stdev:
                    for j in range(num_colours):
                        S = img_data[start+num_nodes*(6+num_colours+j):start+num_nodes*(7+num_colours+j)]
                        S = S / 255  # Normalization
                        feature_matrix.extend(S)
                
                # Construct node feature matrix throgh reshaping and transposing
                x = torch.tensor(feature_matrix).reshape((num_features, num_nodes)).t().float()
                
                # Create edge COO sparse matrix
                num_edges = int(img_data[start+num_nodes*(6+2*num_colours)])
                edge_start_idx = start+num_nodes*(6+2*num_colours)+1
                edge_index = img_data.values[edge_start_idx : edge_start_idx + num_edges*2].astype(np.int64)
                edge_index = torch.tensor(edge_index).reshape([2, num_edges])
                
                # Get label
                lbl = torch.tensor(int(img_data[1]))

                # Save tensors in Data object and return    # TODO: Add pos attribute to allow PyG Vision Transforms
                data_list[i] = Data(x=x, y=lbl, edge_index=edge_index)
            
            # Add data list to overall list
            full_data_list.extend(data_list)

            # Print UI Information
            print(f"{filepath} loaded!")
        
        # Collate data into one massive Data object and save its state
        self.save(full_data_list, self.processed_paths[0])
        
        # Print separator lines
        print('-' * 20)


def adjacent(c1, c2) -> bool:
    """
    A function that takes two cuboids with four features each:
        1) x_centre
        2) y_centre
        3) width
        4) height
    and determines whether the two cuboids are adjacent in x-y
    cartesian space.

    Args:
    - c1: Tensor
        - A tensor of shape (5)
    - c2: Tensor
        - A tensor of shape (5) 
    """
    x_dist = abs(c1[0] - c2[0])
    y_dist = abs(c1[1] - c2[1])
    return (x_dist <= (c1[2] + c2[2])/2) and (y_dist <= (c1[3] + c2[3])/2)


def cupid_to_graph(arg):
    i, dataset, ablation_code = arg

    # Current CuPID data
    cupid_table, label = dataset[i]

    # Filter table to only contain leaf cuboids
    cupid_table = cupid_table[cupid_table["order"] < 0].sort_values(by="order", ascending=False)
    no_of_nodes = len(cupid_table)

    # Fill in feature matrix
    x = torch.zeros((no_of_nodes, len(ablation_code)))
    for i in range(len(cupid_table)):
        row = cupid_table.iloc[i]
        for j in range(len(ablation_code)):
            if ablation_code[j] == 'X':
                x[i][j] = (row["size"][1]+1)/2 + row["start"][1] + 1
                x[i][j] /= 28
            elif ablation_code[j] == 'Y':
                x[i][j] = (row["size"][0]+1)/2 + row["start"][0] + 1
                x[i][j] /= 28
            elif ablation_code[j] == 'C':
                x[i][j] =  round(row["mu"].item())
                x[i][j] /= 255
            elif ablation_code[j] == 'N':
                x[i][j] =  row["n"]
                x[i][j] /= (28*28)
            elif ablation_code[j] == 'A':
                x[i][j] = round(math.atan(row["size"][0]/row["size"][1])*180/math.pi, 2)
                x[i][j] /= 90
            elif ablation_code[j] == 'W':
                x[i][j] = row["size"][1]
                x[i][j] /= 28
            elif ablation_code[j] == 'H':
                x[i][j] = row["size"][0]
                x[i][j] /= 28
    x = x.float()

    # Find edges in COO format
    coo_src = []
    coo_dst = []
    for i in range(len(cupid_table)):
        for j in range(len(cupid_table)):
            if i == j:
                continue
            row1 = cupid_table.iloc[i]
            row2 = cupid_table.iloc[j]

            # Construct cuboid's x_centre, y_centre, width and height
            c1 = ((row1["size"][1]+1)/2 + row1["start"][1] + 1,
                (row1["size"][0]+1)/2 + row1["start"][0] + 1,
                row1["size"][1],
                row1["size"][0]
                )

            c2 = ((row2["size"][1]+1)/2 + row2["start"][1] + 1,
                (row2["size"][0]+1)/2 + row2["start"][0] + 1,
                row2["size"][1],
                row2["size"][0]
                )
            
            if adjacent(c1, c2):
                coo_src.append(-1*row2["order"] - 1)
                coo_dst.append(-1*row1["order"] - 1)
    edge_index = torch.tensor([coo_src, coo_dst], dtype=torch.int64)

    # Create Data object
    label = torch.tensor(int(label))
    return Data(x=x, y=label, edge_index=edge_index)