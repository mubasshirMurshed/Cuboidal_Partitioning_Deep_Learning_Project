# Imports
import torch
from torch import Tensor
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.data import Data
import pandas as pd
from tqdm import tqdm
from numpy import int64
import os
from typing import Optional, Union
from transforms import CupidPartition
from torchvision.datasets import MNIST
from torch.utils.data import random_split


class MNISTGraphDataset(InMemoryDataset):
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
                       train_val_test: int=0,
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
        self.root = root + "mnist" + self.num_cuboids + "/"         # The dataset processed files will be saved here
        self.source_root = root + "mnistPytorch/"                   # The path to MNIST dataset provided by PyTorch
        self.mode = mode
        self.train_val_test = train_val_test
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
        self.data, self.slices = torch.load(self.processed_paths[self.train_val_test])


    @property
    def processed_file_names(self):
        """
        The name of the file which has the processed and saved data.
        """
        return [f'mnistTrain-{self.mode}-{self.num_cuboids}-{self.train_length}-{self.ablation_code}.pt', 
                f'mnistValidation-{self.mode}-{self.num_cuboids}-{self.val_length}-{self.ablation_code}.pt',
                f'mnistTest-{self.mode}-{self.num_cuboids}-{self.test_length}-{self.ablation_code}.pt']


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
        self.dataset = MNIST(root=self.source_root, train=self.train_val_test in [0, 1], transform=CupidPartition(self.num_cuboids), download=True)

        # Deal with train and validation split
        if self.train_val_test in [0, 1]:
            generator = torch.Generator().manual_seed(42)
            self.train, self.val = random_split(self.dataset, [self.train_length, self.val_length], generator=generator)
            if self.train_val_test == 0:
                self.dataset = self.train
            else:
                self.dataset = self.val

        # Make space for data objects from this file
        dataset = [0]*(len(self.dataset))

        # Convert each CuPID image into a graph
        for i in tqdm(range(len(self.dataset)), leave=False):
            # Current CuPID data
            cupid_data, label = self.dataset[i]

            # TODO: Calculate all required info from ablation code desired


            # Create Data object
            # dataset[i] = Data(x=x, y=label, edge_index=edge_index)

        # Collate data into one massive Data object and save its state
        data, slices = self.collate(dataset)
        torch.save((data, slices), self.processed_paths[self.train_val_test])

        # Print UI Information
        print(f"Dataset loaded!")
        
        # Print separator lines
        print('-' * 20)


class MNISTGraphDataset_V6(InMemoryDataset):
    """
    In memory dataset class that stores the dataset in a .pt file under
    root/processed.

    The class will ignore information if it detects that the .pt file is already
    created by specifying its name in the processed_file_names attribute.

    Selection of which feature data to include can be done via flags. By default, every value is
    included.
    """
    def __init__(self, root: Optional[str]=None, 
                       name: Optional[str]=None, 
                       mode: Optional[str]=None, 
                       partition_limit: Optional[int]=None, 
                       length: Optional[int]=None,
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
        - length: int | None
            - Length of dataset to consider
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
        self.root = root
        self.length = length if length is not None else 10000
        self.mode = mode
        self.partition_limit = partition_limit
        self.name = name
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

        # Run inherited processes to create dataset .pt files
        super().__init__(root)

        # Load dataset as self.data
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        """
        List of the raw file names to process and apply transforms. In this case
        our data is already given as csv files.
        """
        directory = os.listdir(f"{self.root}/raw")
        filtered = filter(lambda file: self.name in file and self.mode in file and str(self.partition_limit) in file, directory)
        return list(filtered)


    @property
    def processed_file_names(self):
        """
        The name of the file which has the processed and saved data.
        """
        return [f'{self.name}-{self.mode}-{self.partition_limit}-{self.length*len(self.raw_file_names)}-{self.ablation_code}.pt']


    def process(self):
        """
        Reads in the data from the given filename and loads it all in an array of Data objects
        which is then collated to save easy in a .pt file.

        This features data extraction from the csv file,, normalization of the values, and the creation
        of the Data object for each image, including its feature matrix, adjacency matrix and label.
        """
        # Give UI information
        print("Loading in dataset in memory...")
        full_data_list = []

        for filepath in self.raw_paths:
            # Find max number of columns in csv
            f = open(filepath)
            num_cols = max(len(line.split(',')) for line in f)
            f.close()

            # Read in data file and save attributes
            if self.length is not None:
                df = pd.read_csv(filepath, nrows=self.length, header=None, skiprows=[0], names=range(num_cols)).fillna(0)
            else:
                df = pd.read_csv(filepath, header=None, skiprows=[0], names=range(num_cols)).fillna(0)

            # Make space for data objects from this file
            data_list = [0]*(len(df))

            # Create each Data object and store it
            for i in tqdm(range(len(df)), leave=False):
                # Current Image
                img_data = df.iloc[i]

                # Create node feature matrix
                num_nodes = int(img_data[3])

                # Create the single row vector of relevant values
                feature_matrix = []
                if self.x_centre:
                    X = img_data.values[4:4+num_nodes]
                    X = X / 28  # Normalization
                    feature_matrix.extend(X)
                if self.y_centre:
                    Y = img_data.values[4+num_nodes:4+num_nodes*2]
                    Y = Y / 28  # Normalization
                    feature_matrix.extend(Y)
                if self.colour:
                    C = img_data.values[4+num_nodes*2:4+num_nodes*3]
                    C = C / 255  # Normalization
                    feature_matrix.extend(C)
                if self.num_pixels:
                    N = img_data.values[4+num_nodes*3:4+num_nodes*4]
                    N = N / (28*28)  # Normalization
                    feature_matrix.extend(N)
                if self.angle:
                    A = img_data.values[4+num_nodes*4:4+num_nodes*5]
                    A = A / 90  # Normalization
                    feature_matrix.extend(A)
                if self.width:
                    W = img_data.values[4+num_nodes*5:4+num_nodes*6]
                    W = W / 28  # Normalization
                    feature_matrix.extend(W)
                if self.width:
                    H = img_data.values[4+num_nodes*6:4+num_nodes*7]
                    H = H / 28  # Normalization
                    feature_matrix.extend(H)
                
                x = torch.tensor(feature_matrix).reshape([len(self.ablation_code), num_nodes]).t().float()
                
                # Create edge COO sparse matrix
                num_edges = int(img_data[4 + num_nodes*7])
                edge_start_idx = 5 + num_nodes*7
                edge_index = img_data.values[edge_start_idx : edge_start_idx + num_edges*2].astype(int64)
                edge_index = torch.tensor(edge_index).reshape([2, num_edges])
                edge_index = edge_index - 1         # Correct the indices due to Matlab -> Python
                
                # Get label
                lbl = torch.tensor(int(img_data[1]))

                # Save tensors in Data object and return
                data_list[i] = Data(x=x, y=lbl, edge_index=edge_index)
            
            # Add data list to overall list
            full_data_list.extend(data_list)

            # Print UI Information
            print(f"{filepath} loaded!")
        
        # Collate data into one massive Data object and save its state
        data, slices = self.collate(full_data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        # Print separator lines
        print('-' * 20)


class MNISTGraphDataset_V5(InMemoryDataset):
    """
    In memory dataset class that stores the dataset in a .pt file under
    root/processed.

    The class will ignore information if it detects that the .pt file is already
    created by specifying its name in the processed_file_names attribute.
    """
    def __init__(self, root: Optional[str]=None, 
                       name: Optional[str]=None, 
                       mode: Optional[str]=None, 
                       partition_limit: Optional[int]=None, 
                       length: Optional[int]=None,
                       colour: bool=True,
                       x_centre: bool=True,
                       y_centre: bool=True,
                       num_pixels: bool=True,
                       angle: bool=True) -> None:
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
        - length: int | None
            - Length of dataset to consider
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
        """
        # Save attributes
        self.root = root
        self.length = length if length is not None else 10000
        self.mode = mode
        self.partition_limit = partition_limit
        self.name = name
        self.colour = colour
        self.x_centre = x_centre
        self.y_centre = y_centre
        self.num_pixels = num_pixels
        self.angle = angle

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

        # Run inherited processes to create dataset .pt files
        super(MNISTGraphDataset_V5, self).__init__(root)

        # Load dataset as self.data
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        """
        List of the raw file names to process and apply transforms. In this case
        our data is already given as csv files.
        """
        directory = os.listdir(f"{self.root}/raw")
        filtered = filter(lambda file: self.name in file and self.mode in file and str(self.partition_limit) in file, directory)
        return list(filtered)


    @property
    def processed_file_names(self):
        """
        The name of the file which has the processed and saved data.
        """
        return [f'{self.name}-{self.mode}-{self.partition_limit}-{self.length*len(self.raw_file_names)}-{self.ablation_code}.pt']


    def process(self):
        """
        Reads in the data from the given filename and loads it all in an array of Data objects
        which is then collated to save easy in a .pt file.

        This features data extraction from the csv file,, normalization of the values, and the creation
        of the Data object for each image, including its feature matrix, adjacency matrix and label.
        """
        # Give UI information
        print("Loading in dataset in memory...")
        full_data_list = []

        for filepath in self.raw_paths:
            # Find max number of columns in csv
            f = open(filepath)
            num_cols = max(len(line.split(',')) for line in f)
            f.close()

            # Read in data file and save attributes
            if self.length is not None:
                df = pd.read_csv(filepath, nrows=self.length, header=None, skiprows=[0], names=range(num_cols)).fillna(0)
            else:
                df = pd.read_csv(filepath, header=None, skiprows=[0], names=range(num_cols)).fillna(0)

            # Make space for data objects from this file
            data_list = [0]*(len(df))

            # Create each Data object and store it
            for i in tqdm(range(len(df)), leave=False):
                # Current Image
                img_data = df.iloc[i]

                # Create node feature matrix
                num_nodes = int(img_data[3])

                # Create the single row vector of relevant values
                feature_matrix = []
                if self.x_centre:
                    X = img_data.values[4:4+num_nodes]
                    X = X / 28  # Normalization
                    feature_matrix.extend(X)
                if self.y_centre:
                    Y = img_data.values[4+num_nodes:4+num_nodes*2]
                    Y = Y / 28  # Normalization
                    feature_matrix.extend(Y)
                if self.colour:
                    C = img_data.values[4+num_nodes*2:4+num_nodes*3]
                    C = C / 255  # Normalization
                    feature_matrix.extend(C)
                if self.num_pixels:
                    N = img_data.values[4+num_nodes*3:4+num_nodes*4]
                    N = N / (28*28)  # Normalization
                    feature_matrix.extend(N)
                if self.angle:
                    A = img_data.values[4+num_nodes*4:4+num_nodes*5]
                    A = A / 90  # Normalization
                    feature_matrix.extend(A)
                
                x = torch.tensor(feature_matrix).reshape([len(self.ablation_code), num_nodes]).t().float()
                
                # Create edge COO sparse matrix
                num_edges = int(img_data[4 + num_nodes*5])
                edge_start_idx = 5 + num_nodes*5
                edge_index = img_data.values[edge_start_idx : edge_start_idx + num_edges*2].astype(int64)
                edge_index = torch.tensor(edge_index).reshape([2, num_edges])
                edge_index = edge_index - 1         # Correct the indices due to Matlab -> Python
                
                # Get label
                lbl = torch.tensor(int(img_data[1]))

                # Save tensors in Data object and return
                data_list[i] = Data(x=x, y=lbl, edge_index=edge_index)
            
            # Add data list to overall list
            full_data_list.extend(data_list)

            # Print UI Information
            print(f"{filepath} loaded!")
        
        # Collate data into one massive Data object and save its state
        data, slices = self.collate(full_data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        # Print separator lines
        print('-' * 20)


class MNISTGraphDataset_V4(InMemoryDataset):
    """
    In memory dataset class that stores the dataset in a .pt file under
    root/processed.

    The class will ignore information if it detects that the .pt file is already
    created by specifying its name in the processed_file_names attribute.
    """
    def __init__(self, root: str, mode: str, partition_limit: int, length: Union[int, None]=None, name: str=None) -> None:
        """
        Saves attributes and runs super init to do processing and loading of the data in
        self.data.

        Args:
        - root: str
            - The string path to the root folder where the data files are
        - mode: str
            - The type of dataset being loaded, CP (cuboidal partition) or SP
              (superpixel partition)
        - partition_limit: int
            - The upper limit of number of segments allowed for each image
        - length: int | None
            - Length of dataset to consider
        - train: bool
            - Whether the dataset is the training or validation dataset
        """
        self.root = root
        self.length = length if length is not None else 10000
        self.mode = mode
        self.partition_limit = partition_limit
        self.name = name

        super(MNISTGraphDataset_V4, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        """
        List of the raw file names to process and apply transforms. In this case
        our data is already given as csv files.
        """
        directory = os.listdir(f"{self.root}/raw")
        filtered = filter(lambda file: self.name in file and self.mode in file and str(self.partition_limit) in file, directory)
        return list(filtered)


    @property
    def processed_file_names(self):
        """
        The name of the file which has the processed and saved data.
        """
        return [f'{self.name}-{self.mode}-{self.partition_limit}-length={self.length*len(self.raw_file_names)}.pt']


    def process(self):
        """
        Reads in the data from the given filename and loads it all in an array of Data objects
        which is then collated to save easy in a .pt file.

        This features data extraction from the csv file,, normalization of the values, and the creation
        of the Data object for each image, including its feature matrix, adjacency matrix and label.
        """
        # Give UI information
        print("Loading in dataset in memory...")
        full_data_list = []

        for filepath in self.raw_paths:
            # Find max number of columns in csv
            f = open(filepath)
            num_cols = max(len(line.split(',')) for line in f)
            f.close()

            # Read in data file and save attributes
            if self.length is not None:
                df = pd.read_csv(filepath, nrows=self.length, header=None, skiprows=[0], names=range(num_cols)).fillna(0).astype(int64)
            else:
                df = pd.read_csv(filepath, header=None, skiprows=[0], names=range(num_cols)).fillna(0).astype(int64)

            # Make space for data objects from this file
            data_list = [0]*(len(df))

            # Create each Data object and store it
            for i in tqdm(range(len(df)), leave=False):
                # Current Image
                img_data = df.iloc[i]

                # Create node feature matrix
                num_nodes = img_data[3]
                x = torch.tensor(img_data.values[4:4+num_nodes*3]).reshape([3, num_nodes]).t().float()
                
                # Normalize feature matrix
                x[:, 0] /= 28
                x[:, 1] /= 28
                x[:, 2] /= 255
                
                # Create edge COO sparse matrix
                num_edges = img_data[4 + num_nodes*4]
                edge_start_idx = 5 + num_nodes*4
                edge_index = img_data.values[edge_start_idx : edge_start_idx + num_edges*2]
                edge_index = torch.tensor(edge_index).reshape([2, num_edges])
                edge_index = edge_index - 1         # Correct the indices due to Matlab -> Python
                
                # Get label
                lbl = torch.tensor(img_data[1])

                # Save tensors in Data object and return
                data_list[i] = Data(x=x, y=lbl, edge_index=edge_index)
            
            # Add data list to overall list
            full_data_list.extend(data_list)

            # Print UI Information
            print(f"{filepath} loaded!")
        
        # Collate data into one massive Data object and save its state
        data, slices = self.collate(full_data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        # Print separator lines
        print('-' * 20)


class MNISTGraphDataset_V3(InMemoryDataset):
    """
    In memory dataset class that stores the dataset in a .pt file under
    root/processed.

    The class will ignore information if it detects that the .pt file is already
    created by specifying its name in the processed_file_names attribute.
    """
    def __init__(self, root: str, filename: str, transform=None, pre_transform=None, length=None) -> None:
        """
        Saves attributes and runs super init to do processing and loading of the data in
        self.data.

        Args:
        - root: str
            - The string path to the root folder where the data files are
        - filename: str
            - The filename of the data file in the root folder to process
        - transform:
            - Transforms applied to the data whilst obtaining the data via get()
        - pre_transform:
            - Transforms applied to the data when processing and saving
        - length:
            - Length of dataset to consider
        """
        self.length = length
        self.filename = filename
        self.file_root = f"{root}/{filename}.csv" 
        super(MNISTGraphDataset_V3, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        """
        List of the raw file names to process and apply transforms. In this case
        our data is already given as csv files.
        """
        return []


    @property
    def processed_file_names(self):
        """
        The name of the file which has the processed and saved data.
        """
        return [f'{self.filename}.pt']


    def process(self):
        """
        Reads in the data from the given filename and loads it all in an array of Data objects
        which is then collated to save easy in a .pt file.

        This features data extraction from the csv file,, normalization of the values, and the creation
        of the Data object for each image, including its feature matrix, adjacency matrix and label.
        """
        # Give UI information
        print("Loading in dataset in memory...")

        # Read in data file and save attributes
        if self.length is not None:
            df = pd.read_csv(self.file_root, nrows=self.length, header=None, skiprows=[0]).fillna(0).astype(int64)
        else:
            df = pd.read_csv(self.file_root, header=None, skiprows=[0]).fillna(0).astype(int64)

        # Fill in self.dataset with the appropriate reshapes of the data
        data_list = [0]*len(df)
        for i in tqdm(range(len(df)), leave=False):
            # Current Image
            img_data = df.iloc[i]

            # Create node feature matrix
            num_nodes = img_data[3]
            x = torch.tensor(img_data.values[4:4+num_nodes*3]).reshape([3, num_nodes]).t().float()
            
            # Normalize feature matrix
            x[:, 0] /= 28
            x[:, 1] /= 28
            x[:, 2] /= 255
            
            # Create edge COO sparse matrix
            num_edges = img_data[4 + num_nodes*4]
            edge_start_idx = 5 + num_nodes*4
            edge_index = img_data.values[edge_start_idx : edge_start_idx + num_edges*2]
            edge_index = torch.tensor(edge_index).reshape([2, num_edges])
            edge_index = edge_index - 1         # Correct the indices due to Matlab -> Python
            
            # Get label
            lbl = torch.tensor(img_data[1])

            # Save tensors in Data object and return
            data_list[i] = Data(x=x, y=lbl, edge_index=edge_index)
        
        # Apply filters + transforms
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        # Collate data into one massive Data object and save its state
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        # Print separator lines
        print('-' * 20)


class MNISTGraphDataset_V2(InMemoryDataset):
    """
    In memory dataset class that stores the dataset in a .pt file under
    root/processed.

    The class will ignore information if it detects that the .pt file is already
    created by specifying its name in the processed_file_names attribute.
    """
    def __init__(self, root: str, filename: str, transform=None, pre_transform=None, length=None) -> None:
        """
        Saves attributes and runs super init to do processing and loading of the data in
        self.data.

        Args:
        - root: str
            - The string path to the root folder where the data files are
        - filename: str
            - The filename of the data file in the root folder to process
        - transform:
            - Transforms applied to the data whilst obtaining the data via get()
        - pre_transform:
            - Transforms applied to the data when processing and saving
        - length:
            - Length of dataset to consider
        """
        self.length = length
        self.filename = filename
        self.file_root = f"{root}/{filename}.csv" 
        super(MNISTGraphDataset_V2, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        """
        List of the raw file names to process and apply transforms. In this case
        our data is already given as csv files.
        """
        return []


    @property
    def processed_file_names(self):
        """
        The name of the file which has the processed and saved data.
        """
        return [f'{self.filename}.pt']


    def process(self):
        """
        Reads in the data from the given filename and loads it all in an array of Data objects
        which is then collated to save easy in a .pt file.

        This features data extraction from thr csv file,, normalization of the values, and the creation
        of the Data object for each image, including its feature matrix, adjacency matrix and label.
        """
        # Give UI information
        print("Loading in dataset in memory...")

        # Read in data file and save attributes
        if self.length is not None:
            df = pd.read_csv(self.file_root, nrows=self.length)
        else:
            df = pd.read_csv(self.file_root)

        # Fill in self.dataset with the appropriate reshapes of the data
        data_list = [0]*len(df)
        for i in tqdm(range(len(df)), leave=False):
            # Current Image
            img_data = df.iloc[i]

            # Create space for node feature matrix
            num_cuboids = img_data[0]
            x = torch.zeros(num_cuboids, 5)
            
            # Save feature matrix
            for j in range(num_cuboids):
                # Obtain feature vector data for cuboid j
                x_pos = img_data[5*j + 1]
                y_pos = img_data[5*j + 2]
                width = img_data[5*j + 3]
                height = img_data[5*j + 4]
                colour = img_data[5*j + 5]
                x_centre = x_pos + width/2
                y_centre = y_pos + height/2

                # Save feature vector for cuboid j
                x[j][0] = x_centre
                x[j][1] = y_centre
                x[j][2] = colour
                x[j][3] = width
                x[j][4] = height
            
            # Create space for edge connections
            adj_matrix = torch.zeros(num_cuboids, num_cuboids)
            num_edges = 0
            for m in range(num_cuboids):
                for n in range(m + 1, num_cuboids):
                    if adjacent(x[m], x[n]):
                        num_edges += 1
                        adj_matrix[m][n] = 1
                        adj_matrix[n][m] = 1
            
            # Convert adjacency matrix to edge tensor in COO format
            edges = adj_matrix.nonzero().t().contiguous()
            
            # Normalize feature matrix
            for j in range(num_cuboids):
                x[j][0] /= 28   # Normalize x_centre
                x[j][1] /= 28   # Normalize y_centre
                x[j][2] /= 255  # Normalize gray colour
                x[j][3] /= 28   # Normalize widths
                x[j][4] /= 28   # Normalize heights

            # Get label
            lbl = img_data[-2]

            # Save tensors in Data object and return
            data_list[i] = Data(x=x[:, :3], y=lbl, edge_index=edges)
        
        # Apply filters + transforms
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        # Collate data into one massive Data object and save its state
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        # Print separator lines
        print('-' * 20)


class MNISTGraphDataset_V1(Dataset):
    def __init__(self, csv_file_dir, length = None) -> None:
        # Give UI information
        print("Loading in dataset in memory...")

        # Read in data file and save attributes
        if length is not None:
            self.data = pd.read_csv(csv_file_dir, nrows=length)
        else:
            self.data = pd.read_csv(csv_file_dir)

        # Fill in self.dataset with the appropriate reshapes of the data
        self.dataset = [0]*len(self.data)
        for i in tqdm(range(len(self.data)), leave=False):
            # Current Image
            img_data = self.data.values[i]

            # Create space for node feature matrix
            num_cuboids = img_data[0]
            x = torch.zeros(num_cuboids, 5)

            # # Create space for node position matrix
            # pos = torch.zeros(num_cuboids, 2, dtype=torch.long)
            
            # Save feature matrix
            for j in range(num_cuboids):
                # Obtain feature vector data for cuboid j
                x_pos = img_data[5*j + 1]
                y_pos = img_data[5*j + 2]
                width = img_data[5*j + 3]
                height = img_data[5*j + 4]
                colour = img_data[5*j + 5]
                x_centre = x_pos + width/2
                y_centre = y_pos + height/2

                # Save feature vector for cuboid j
                x[j][0] = x_centre
                x[j][1] = y_centre
                x[j][2] = colour
                x[j][3] = width
                x[j][4] = height
            
            # Create space for edge connections
            adj_matrix = torch.zeros(num_cuboids, num_cuboids)
            num_edges = 0
            for m in range(num_cuboids):
                for n in range(m + 1, num_cuboids):
                    if adjacent(x[m], x[n]):
                        num_edges += 1
                        adj_matrix[m][n] = 1
            
            # Convert adjacency matrix to edge tensor in COO format
            edges = torch.zeros(2, 2*num_edges, dtype=torch.long)
            pointer = 0
            for m in range(num_cuboids):
                for n in range(m + 1, num_cuboids):
                    if adj_matrix[m][n]:
                        # Store edge
                        edges[0][pointer] = m  # Provide source
                        edges[1][pointer] = n  # Provide dest

                        # Update pointer to store reverse edge
                        pointer += 1
                        edges[0][pointer] = n  # Need both directions
                        edges[1][pointer] = m

                        # Update pointer for next edge
                        pointer += 1
            
            # Normalize feature matrix
            for j in range(num_cuboids):
                x[j][0] /= 28   # Normalize x_centre
                x[j][1] /= 28   # Normalize y_centre
                x[j][2] /= 255  # Normalize gray colour
                x[j][3] /= 28   # Normalize widths
                x[j][4] /= 28   # Normalize heights

            # Get label
            lbl = torch.tensor(img_data[-2])

            # Save tensors in Data object and return
            self.dataset[i] = Data(x=x[:, :3], y=lbl, edge_index=edges)
        
        # Print separator lines
        print('-' * 20)

    def __len__(self):
        """
        Return length of the dataset.
        """
        return len(self.data)

    def get(self, index: int):
        """
        Retrieves partitioning data of a selected image and its label by accessing internally
        stored data.

        Args:
        - index: int
            - Index of graph
        """
        return self.dataset[index]
    

def adjacent(c1: Tensor, c2: Tensor) -> bool:
    """
    A function that takes two cuboids with five features each:
        1) x_centre
        2) y_centre
        3) colour
        4) width
        5) height
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
    return (x_dist <= (c1[3] + c2[3])/2) and (y_dist <= (c1[4] + c2[4])/2)