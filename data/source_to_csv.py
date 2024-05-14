import os
from transforms import CuPIDPartition
from data_classes import MyMNIST, MyCIFAR_10, MyMedMNIST, MyOmniglot
from typing import Optional
import csv
import math
from tqdm import tqdm
import multiprocessing as mp
import numpy as np

class CSV_Dataset_Creator:
    def __init__(self, root: Optional[str], dataset: str, num_segments: Optional[int]=None, mode: Optional[str]=None, max_entries_per_file:int=10000, chunksize=200) -> None:
        # Save attributes and path information
        self.num_segments = num_segments
        self.data_source = root + dataset                                           # The path to dataset provided
        self.root = root + dataset + "/" + f"{self.num_segments}" + "/raw/"         # The dataset processed files will be saved here
        self.chunksize = chunksize
        
        # Partitioning mode (CP/SP)
        self.mode = mode
        self.max_entries_per_file = max_entries_per_file
        self.dataset_name = dataset

        # Determine partitioning transform
        if self.mode == "CP":
            transform = CuPIDPartition(self.num_segments)

        # Data Source with Partition transform
        match dataset:
            case "mnist":
                self.dataset = MyMNIST(transform)
            case "cifar":
                self.dataset = MyCIFAR_10(transform)
            case "medmnist":
                self.dataset = MyMedMNIST(transform)
            case "omniglot":
                self.dataset = MyOmniglot(transform)
            case _:
                print("Error, dataset name not recognised or supported.")
                return

        # Get splits
        self.train_dataset = self.dataset.train()
        self.val_dataset = self.dataset.validation()
        self.test_dataset = self.dataset.test()

    def create_csv_files(self, verbose:bool=False) -> None:
        self.create_csv_file(self.train_dataset, "Train", verbose)
        self.create_csv_file(self.val_dataset, "Validation", verbose)
        self.create_csv_file(self.test_dataset, "Test", verbose)

    def create_csv_file(self, dataset, split: str, verbose: bool=False) -> None:
        # Filename of where to write
        filepath = f"{self.root}{self.dataset_name}{split}-{self.mode}-{self.num_segments}-1.csv"

        # Check that the directory exists and if not, create it
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # If file already exists, then return
        if os.path.isfile(filepath):
            if verbose: print(f"{split} dataset already exists.")
            return
        
        # Create multiprocessing pool
        with mp.Pool(processes=24) as pool:
            if verbose: print(f"Generating {split} dataset files...")
            
            # Iterate over every chunk of data
            for i in tqdm(range(math.ceil(len(dataset) / self.max_entries_per_file)), position=0, leave=False, disable=not verbose):
                # For each image in the chunk, pre allocate space for data collection
                start = i*self.max_entries_per_file
                end = min((i+1)*self.max_entries_per_file, len(dataset))

                # Obtain the relevant CuPID data of all images in this chunk in parallel
                data = list(tqdm(pool.imap(obtain_cupid_data, zip(range(start, end), [dataset]*(end - start)), self.chunksize), total=end-start, position=1, leave=False, disable=not verbose))               

                # File writing
                filepath = f"{self.root}{self.dataset_name}{split}-{self.mode}-{self.num_segments}-{i+1}.csv"
                file = open(filepath, 'w', newline='')
                writer = csv.writer(file)
                writer.writerow(["Sample_No", "Label", "PSNR", "No_of_Nodes", "No_of_Features", "Length_X", "Length_Y", 
                                 "Centre_Xs", "Centre_Ys", "Values", "No_of_Pixels", "BoxAngles", "BoxWidths", "BoxHeights", "Standard_Deviations", "No_of_edges", "COO_src", "COO_dst"])

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
    # Obtain CuPID data of the image
    i, dataset = arg
    cupid_table, label = dataset[i]
    if type(label) != int:
        label = np.array(label).item()

    # Filter table to only contain leaf cuboids
    cupid_table = cupid_table[cupid_table["order"] < 0].sort_values(by="order", ascending=False)
    
    # Initialise first few column values
    shape = dataset.shape
    no_of_nodes = len(cupid_table)
    no_of_features = shape[-1]
    new_row_entry = [i, label, 0, no_of_nodes, no_of_features, shape[0], shape[1]]

    # Add in centre x coordinates cuboid
    for _, row in cupid_table.iterrows():
        new_row_entry.append((row["size"][1]+1)/2 + row["start"][1] + 1)

    # Add in centre y coordinates for each cuboid
    for _, row in cupid_table.iterrows():
        new_row_entry.append((row["size"][0]+1)/2 + row["start"][0] + 1)

    # Add in colour values for each cuboid (restore float to be between 0-255)
    for j in range(no_of_features):
        for _, row in cupid_table.iterrows():
            new_row_entry.append(np.around(row["mu"][j]*255, 2))
    
    # Add in num of pixels for each cuboid
    for _, row in cupid_table.iterrows():
        new_row_entry.append(row["n"])

    # Add in box angles for each cuboid
    for _, row in cupid_table.iterrows():
        new_row_entry.append(round(math.atan(row["size"][0]/row["size"][1])*180/math.pi, 2))        # This is hardcoded to 2D

    # Add in box width for each cuboid
    for _, row in cupid_table.iterrows():
        new_row_entry.append(row["size"][1])

    # Add in box height for each cuboid
    for _, row in cupid_table.iterrows():
        new_row_entry.append(row["size"][0])

    # Add in standard deviation for each cuboid w.r.t original sections (scaled to 255 colour space)
    for j in range(no_of_features):
        for _, row in cupid_table.iterrows():
            new_row_entry.append(np.around(np.sqrt(row["sigma2"][j])*255, 2))

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
                coo_src.append(-1*row2["order"] - 1)
                coo_dst.append(-1*row1["order"] - 1)

    # Add the edge information
    new_row_entry.append(no_of_edges)
    new_row_entry.extend(coo_src)
    new_row_entry.extend(coo_dst)

    return new_row_entry

import time

def main():
    creator = CSV_Dataset_Creator("data/csv/", "mnist", 8, "CP", chunksize=1)
    start = time.perf_counter()
    creator.create_csv_files(verbose=True)
    end = time.perf_counter()
    return end - start

if __name__ == '__main__':
    t = main()
    print(t)