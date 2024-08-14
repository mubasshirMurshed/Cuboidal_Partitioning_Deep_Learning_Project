import os
from transforms import CuPIDPartition, SLICPartition
from data_classes import SourceDataset, MyMNIST, MyCIFAR_10, MyMedMNIST, MyOmniglot
import csv
import math
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Type
from enums import Split, Partition

# TODO: Add replace/overwiter functionality, which skips the exist check and overwrites anyway.
# TODO: Throw errors instead of returning for the error cases
# TODO: Change processes to num_workers argument able to passed
# TODO: Put partition_to_graph function as partition method obtained from inheritance
# TODO: Move file writing out of mp pool

"""
A class that is responsible for creating the multiple csv files containing the partitioned data
for each split. This means a csv file will be created for each split and can be created based on
the partitioning scheme. The point is to save on partitioning processing time for neural network
training. Only 2D images are supported. Any feature length is supported. 

Each row in a csv file contains the information of 1 sample. The columns follow a specific reading format.
First, the ID and label of the data is written. Then the PSNR is recorded, followed by the number of segments
created, the number of features of the data (RGB = 3 features). Then the shape of the data. Then if N = number 
of segments, it is followed by N number of data for various pieces of data, where each value corresponds to a
particular segment. Finally, the number of edges in the adj matrix is encoded followed by the edges in COO format. 

If the first instance of a csv file is detected for a given split and partition, it will stop
and assume the files already exist.
"""
class CSV_Dataset_Creator:
    def __init__(self, root: str, dataset: Type[SourceDataset], num_segments: int, mode: Partition, max_entries_per_file: int=10000, chunksize: int=200) -> None:
        """
        Saves attributes, and creates necessary extra data based on mode selected and dataset. Creates links to the raw splits of 
        the selected datasets with the appropriate partitioning algorithm provided as a transform.

        Args:
        - root: str
            - The root directory of where the csv files should be stored. This will be prepended with subdirectories involving
              the dataset name and the number of segments
        - dataset: SourceDataset
            - Class constructor for a raw dataset in data/source/
        - num_segments: int
            - How many clusters to partition the data into
        - mode: Partition
            - Partitioning mode
        - max_entries_per_file: int
            - A variable to control the number of entries per csv file created. For example if the training dataset contained 20000 samples
              and you set this variable to 1000, it would result in 20 csv files being created for that dataset.
        - chunksize: int
            - A variable to control the parallelism of splitting up the pre-processing amongst several cores. Set this to higher values
              for more intense datasets/partitions.
        """
        # Save attributes and path information
        self.num_segments = num_segments
        self.dataset_name = dataset.name
        self.data_source = root + dataset.name                                           # The path to dataset provided
        self.root = root + dataset + "/" + f"{self.num_segments}" + "/raw/"         # The dataset processed files will be saved here
        self.chunksize = chunksize
        self.max_entries_per_file = max_entries_per_file
        
        # Determine partitioning transform
        self.mode = mode
        if self.mode is Partition.CuPID:
            transform = CuPIDPartition(self.num_segments)
        elif self.mode is Partition.SLIC:
            transform = SLICPartition(self.num_segments)
        else:
            print("Error, mode is not recognised or supported.")
            return

        # Data Source with Partition transform
        self.dataset = dataset(transform)

        # Get splits
        self.train_dataset = self.dataset.train_dataset()
        self.val_dataset = self.dataset.validation_dataset()
        self.test_dataset = self.dataset.test_dataset()

    def create_csv_files(self, verbose: bool=False) -> None:
        """
        Creates all csv files according to each split.
        """
        self.create_csv_file(self.train_dataset, Split.TRAIN, verbose)
        self.create_csv_file(self.val_dataset, Split.VALIDATION, verbose)
        self.create_csv_file(self.test_dataset, Split.TEST, verbose)

    def create_csv_file(self, dataset: Dataset, split: Split, verbose: bool=False) -> None:
        """
        For a given split and a dataset provided, transform its full contents into partitioned data
        which is then written in a csv file for easy extraction from an in-memory dataset.
        """
        # Filename of where to write
        filepath = f"{self.root}/{self.dataset_name}-{split.value}-{self.mode.value}-{self.num_segments}-1.csv"

        # Check that the directory exists and if not, create it
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # If file already exists, then return
        if os.path.isfile(filepath):
            if verbose: print(f"{split} dataset already exists.")
            return
        
        # Create multiprocessing pool
        with mp.Pool(processes=28) as pool:
            if verbose: print(f"Generating {split} dataset files...")
            
            # Iterate over every chunk of data
            for i in tqdm(range(math.ceil(len(dataset) / self.max_entries_per_file)), position=0, leave=False, disable=not verbose):
                # For each image in the chunk, pre allocate space for data collection
                start = i*self.max_entries_per_file
                end = min((i+1)*self.max_entries_per_file, len(dataset))

                # Obtain the relevant CuPID data of all images in this chunk in parallel
                if self.mode is Partition.CuPID:
                    data = list(tqdm(pool.imap(obtain_cupid_data, zip(range(start, end), [dataset]*(end - start)), self.chunksize), total=end-start, position=1, leave=False, disable=not verbose))               
                elif self.mode is Partition.SLIC:
                    data = list(tqdm(pool.imap(obtain_slic_data, zip(range(start, end), [dataset]*(end - start)), self.chunksize), total=end-start, position=1, leave=False, disable=not verbose)) 

                # File writing
                filepath = f"{self.root}/{self.dataset_name}-{split.value}-{self.mode.value}-{self.num_segments}-{i+1}.csv"
                file = open(filepath, 'w', newline='')
                writer = csv.writer(file)
                writer.writerow(["Sample_No", "Label", "PSNR", "No_of_Nodes", "No_of_Features", "Length_X", "Length_Y", 
                                 "Center_Xs", "Center_Ys", "Values", "No_of_Pixels", "BoxAngles", "BoxWidths", "BoxHeights", 
                                 "Standard_Deviations", "No_of_edges", "COO_src", "COO_dst"])

                # Add all data entries to csv file
                for row in data:
                    writer.writerow(row)

                # csv file complete
                file.close()

        if verbose: print(f"{split} dataset files generated.\n")


def obtain_cupid_data(arg: Tuple[int, Dataset]):
    """
    Given a dataset sample partitioned by CuPID, extract the relevant information to be written
    into a csv file row all into one list. This function is expected to be ran in parallel using
    Python multiprocessing.Pool.imap.
    """
    # Obtain CuPID data of the image
    i, dataset = arg
    cupid_object, label = dataset[i]

    # If label is not an integer (inside an array) extract it
    if type(label) != int:
        label = np.array(label).item()

    # Filter table to only contain leaf cuboids
    cupid_table = cupid_object.cuboids
    cupid_table = cupid_table[cupid_table["order"] < 0].sort_values(by="order", ascending=False)
    
    # Initialise first few column values
    shape = dataset.data_shape
    no_of_nodes = len(cupid_table)
    no_of_features = shape[-1]
    new_row_entry = [i, label, 0, no_of_nodes, no_of_features, shape[0], shape[1]]

    # Add in center x coordinates cuboid
    for _, row in cupid_table.iterrows():
        new_row_entry.append((row["size"][1]+1)/2 + row["start"][1] + 1)

    # Add in center y coordinates for each cuboid
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
        new_row_entry.append(round(math.atan(row["size"][0]/row["size"][1])*180/math.pi, 2))  # This is hardcoded to 2D

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
    adj_matrix = cupid_object.adjacency_matrix()
    no_of_edges = np.sum(adj_matrix)
    coo_src, coo_dst = np.where(adj_matrix)

    # Add the edge information
    new_row_entry.append(no_of_edges)
    new_row_entry.extend(coo_src)
    new_row_entry.extend(coo_dst)

    return new_row_entry

def obtain_slic_data(arg: Tuple[int, Dataset]):
    """
    Given a dataset sample partitioned by SLIC, extract the relevant information to be written
    into a csv file row all into one list. This function is expected to be ran in parallel using
    Python multiprocessing.Pool.imap.
    """
    # Obtain SLIC data of the image
    i, dataset = arg
    slic_object, label = dataset[i]
    if type(label) != int:
        label = np.array(label).item()
    
    # Initialise first few column values
    slic_table = slic_object.segments
    shape = dataset.data_shape
    no_of_nodes = len(slic_table)
    no_of_features = shape[-1]
    new_row_entry = [i, label, 0, no_of_nodes, no_of_features, shape[0], shape[1]]

    # Add in center x coordinates segment
    for _, row in slic_table.iterrows():
        new_row_entry.append(round(row["x_center"], 2))

    # Add in center y coordinates for each segment
    for _, row in slic_table.iterrows():
        new_row_entry.append(round(row["y_center"], 2))

    # Add in colour values for each segment (restore float to be between 0-255)
    for j in range(no_of_features):
        for _, row in slic_table.iterrows():
            new_row_entry.append(np.around(row["mu"][j]*255, 2))
    
    # Add in num of pixels for each segment
    for _, row in slic_table.iterrows():
        new_row_entry.append(row["n"])

    # Add in box angles for each segment
    for _, row in slic_table.iterrows():
        new_row_entry.append(round(math.atan(row["height"]/row["width"])*180/math.pi, 2))        # This is hardcoded to 2D

    # Add in box width for each segment
    for _, row in slic_table.iterrows():
        new_row_entry.append(row["width"])

    # Add in box height for each segment
    for _, row in slic_table.iterrows():
        new_row_entry.append(row["height"])

    # Add in standard deviation for each segment w.r.t original sections (scaled to 255 colour space)
    for j in range(no_of_features):
        for _, row in slic_table.iterrows():
            new_row_entry.append(np.around(np.sqrt(row["sigma2"][j])*255, 2))

    # Calculate number of edges present
    adj_matrix = slic_object.adjacency_matrix()
    no_of_edges = np.sum(adj_matrix)
    coo_src, coo_dst = np.where(adj_matrix)

    # Add the edge information
    new_row_entry.append(no_of_edges)
    new_row_entry.extend(coo_src)
    new_row_entry.extend(coo_dst)

    return new_row_entry


"""
Script to create the CSV files.
"""
import time
def main():
    creator = CSV_Dataset_Creator("data/csv/", MyOmniglot, 128, "SP", chunksize=10)
    start = time.perf_counter()
    creator.create_csv_files(verbose=True)
    end = time.perf_counter()
    return end - start

if __name__ == '__main__':
    t = main()
    print(t)