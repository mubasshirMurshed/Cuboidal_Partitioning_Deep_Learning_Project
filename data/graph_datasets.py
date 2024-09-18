# Imports
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from data.data_classes import SourceDataset
from enums import Split, Partition


class Graph_Dataset(Dataset):
    def __init__(self, dataset: SourceDataset, num_segments: int, x_center: bool=False,
                       y_center: bool=False, colour: bool=False, width: bool=False, height: bool=False, 
                       num_pixels: bool=False, angle: bool=False, st_dev: bool=False) -> None:
        # Set up attributes
        super().__init__()
        self.dataset = dataset
        self.num_segments = num_segments
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


    def len(self) -> int:
        return len(self.dataset)
    
    
    def get(self, idx) -> Data:
        partition_object, label = self.dataset[idx]
        img_data = partition_object.transform_to_csv_data()

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
        lbl = torch.tensor(int(label))

        # Save tensors in Data object and return
        return Data(x=x, y=lbl, edge_index=edge_index)


class Graph_Dataset_CSV(InMemoryDataset):
    """
    In memory dataset class that stores the dataset in a .pt file under
    root/dataset/num_segments/processed. Assumes .csv files exist in
    root/dataset/num_segments/raw.

    The class will ignore information if it detects that the .pt file is already
    created by specifying its name in the processed_file_names attribute.

    Selection of which feature data to include can be done via flags. By default, every value is
    included.
    """
    def __init__(self, root: str, name: str, split: Split, mode: Partition, num_segments: int, length: int | None=None,
                       x_center: bool=False, y_center: bool=False, colour: bool=False, width: bool=False, height: bool=False,
                       num_pixels: bool=False, angle: bool=False, stdev: bool=False) -> None:
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
            - Length of dataset to only load, if None, all will be loaded
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
        self.dataset_size = length
        self.mode = mode.value
        self.num_segments = num_segments
        self.split = split.value

        # Ablation attributes
        self.colour = colour
        self.x_center = x_center
        self.y_center = y_center
        self.num_pixels = num_pixels
        self.angle = angle
        self.width = width
        self.height = height
        self.stdev = stdev

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

        # Get length of dataset to be loaded in
        num_rows = [0]*len(self.raw_paths)
        for i, filepath in enumerate(self.raw_paths):
            # Find number of rows in csv
            f = open(filepath)
            num_rows[i] = sum(1 for _ in f) - 1
            f.close()

        # Set dataset size if None was set
        if self.dataset_size is None or self.dataset_size > sum(num_rows):
            self.dataset_size = sum(num_rows)

        # Determine index of last raw filepath to process and specific number to load to meet length requirement
        total = 0
        self.last_filepath_idx = len(self.raw_paths) - 1
        for i in range(len(self.raw_paths)):
            total += num_rows[i]
            if total >= self.dataset_size:
                self.last_filepath_idx = i
                total -= num_rows[i]
                break
        self.last_file_load_amount = self.dataset_size - total

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
        return [f'{self.name}-{self.split}-{self.mode}-{self.num_segments}-{self.dataset_size}-{self.ablation_code}.pt']


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

        # Iterate through each file until all processed or all requested data is loaded
        for i, filepath in enumerate(self.raw_paths):
            # How much to read based on file idx
            if i < self.last_filepath_idx:
                read_all = True
            elif i == self.last_filepath_idx:
                read_all = False
            else:
                break

            # If requested length is 0 then just exit
            if self.dataset_size == 0:
                break

            # Read in data file and save attributes
            if read_all:
                df = pd.read_csv(filepath, header=None, skiprows=[0], names=range(num_cols[i])).fillna(0)
            else:
                df = pd.read_csv(filepath, nrows=self.last_file_load_amount, header=None, skiprows=[0], names=range(num_cols[i])).fillna(0)

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
