from datasets.transforms import CupidPartition
from typing import Optional
from torchvision.datasets import MNIST
import torch
from torch.utils.data import random_split
import csv
import math
from tqdm import tqdm
import os
import multiprocessing as mp

class rawMNISTDataset:
    def __init__(self, root: Optional[str]=None,
                       num_cuboids: Optional[int]=None,
                       mode: Optional[str]=None, 
                       train_length: int=50000,
                       max_entries_per_file:int=10000) -> None:
        
        # Save attributes and path information
        self.num_cuboids = num_cuboids
        self.root = root + "mnist" + f"{self.num_cuboids}" + "/raw/"        # The dataset processed files will be saved here
        self.source_root = root + "mnistPytorch/"                           # The path to MNIST dataset provided by PyTorch
        self.mode = mode
        self.train_length = train_length
        self.max_entries_per_file = max_entries_per_file

        # MNIST Data Source with CuPID transform
        self.dataset = MNIST(root=self.source_root, train=True, transform=CupidPartition(self.num_cuboids), download=True)
        self.test_dataset = MNIST(root=self.source_root, train=False, transform=CupidPartition(self.num_cuboids), download=True)
        self.val_length = len(self.dataset) - self.train_length
        self.test_length = len(self.test_dataset)

        # Deal with train and validation split
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = random_split(self.dataset, 
                                                            [self.train_length, self.val_length], 
                                                            generator=generator)

    def create_csv_files(self, verbose:bool=False) -> None:
        self.create_csv_file(self.train_dataset, "Train", verbose)
        self.create_csv_file(self.val_dataset, "Validation", verbose)
        self.create_csv_file(self.test_dataset, "Test", verbose)

    def create_csv_file(self, dataset, split: str, verbose: bool=False) -> None:
        # Filename of where to write
        filepath = f"{self.root}mnist{split}-{self.mode}-{self.num_cuboids}-1.csv"

        # Check that the directory exists and if not, create it
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # If file already exists, then return
        if os.path.isfile(filepath):
            if verbose: print(f"{split} dataset already exists.")
            return
        
        # Create multiprocessing pool
        pool = mp.Pool(30)

        if verbose: print(f"Generating {split} dataset files...")\
        # Iterate over every chunk of data
        for i in tqdm(range(math.ceil(len(dataset) / self.max_entries_per_file)), position=0, leave=False, disable=not verbose):
            # For each image in the chunk, pre allocate space for data collection
            start = i*self.max_entries_per_file
            end = min((i+1)*self.max_entries_per_file, len(dataset))

            # Obtain the relevant CuPID data of all images in this chunk in parallel
            data = list(tqdm(pool.imap(obtain_cupid_data, zip(range(start, end), [dataset]*(end - start)), 1), total=end-start, position=1, leave=False, disable=not verbose))               

            # File writing
            filepath = f"{self.root}mnist{split}-{self.mode}-{self.num_cuboids}-{i+1}.csv"
            file = open(filepath, 'w', newline='')
            writer = csv.writer(file)
            writer.writerow(["Sample_No", "Label", "PSNR", "No_of_nodes", "Centre_Xs", "Centre_Ys", "Values", "No_of_Pixels", "BoxAngles", "BoxWidths", "BoxHeights", "No_of_edges", "COO_src", "COO_dst"])

            # Add all data entries to csv file
            for row in data:
                writer.writerow(row)

            # csv file complete
            file.close()

        if verbose: print(f"{split} dataset files generated.\n")


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


# Create function to be ran in parallel
def obtain_cupid_data(arg):
    i, dataset = arg
    # Obtain CuPID data of the image
    cupid_table, label = dataset[i]

    # Filter table to only contain leaf cuboids
    cupid_table = cupid_table[cupid_table["order"] < 0].sort_values(by="order", ascending=False)
    
    # Initialise first few column values
    no_of_nodes = len(cupid_table)
    new_row_entry = [i, label, 0, no_of_nodes]

    # Add in centre x coordinates cuboid
    for _, row in cupid_table.iterrows():
        new_row_entry.append((row["size"][1]+1)/2 + row["start"][1] + 1)

    # Add in centre y coordinates for each cuboid
    for _, row in cupid_table.iterrows():
        new_row_entry.append((row["size"][0]+1)/2 + row["start"][0] + 1)

    # Add in colour values for each cuboid (restore float to integer between 0-255)
    for _, row in cupid_table.iterrows():
        new_row_entry.append(round(row["mu"].item()))
    
    # Add in num. of pixels for each cuboid
    for _, row in cupid_table.iterrows():
        new_row_entry.append(row["n"])

    # Add in box angles for each cuboid
    for _, row in cupid_table.iterrows():
        new_row_entry.append(round(math.atan(row["size"][0]/row["size"][1])*180/math.pi, 2))

    # Add in box width for each cuboid
    for _, row in cupid_table.iterrows():
        new_row_entry.append(row["size"][1])

    # Add in box height for each cuboid
    for _, row in cupid_table.iterrows():
        new_row_entry.append(row["size"][0])

    # Calculate number of edges present
    no_of_edges = 0
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
                no_of_edges += 1
                coo_src.append(-1*row2["order"])
                coo_dst.append(-1*row1["order"])

    # Add the edge information
    new_row_entry.append(no_of_edges)
    new_row_entry.extend(coo_src)
    new_row_entry.extend(coo_dst)

    return new_row_entry