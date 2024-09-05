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


class CSV_Dataset_Writer:
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
    def __init__(self, root: str, dataset: Type[SourceDataset], num_segments: int, mode: Partition, max_entries_per_file: int=10000, 
                 chunksize: int=200, overwrite: bool=False, num_workers: int=28) -> None:
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
        self.dataset_name = dataset.name()
        self.data_source = root + dataset.name()                                           # The path to dataset provided
        self.root = self.data_source + "/" + f"{self.num_segments}" + "/raw/"         # The dataset processed files will be saved here
        self.chunksize = chunksize
        self.max_entries_per_file = max_entries_per_file
        self.overwrite = overwrite
        self.num_workers = num_workers
        
        # Determine partitioning transform
        self.mode = mode
        if self.mode is Partition.CuPID:
            transform = CuPIDPartition(self.num_segments)
        elif self.mode is Partition.SLIC:
            transform = SLICPartition(self.num_segments)
        else:
            raise ValueError(f"Supplied 'mode' argument not a registered partitioning strategy. Got {self.mode} but should be been a Partition Enum.")

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
        # Number of files to make based on each section
        num_files = math.ceil(len(dataset) / self.max_entries_per_file)

        # Filename of where to write
        filepaths = [f"{self.root}/{self.dataset_name}-{split.value}-{self.mode.value}-{self.num_segments}-{i+1}.csv" for i in range(num_files)]

        # If all exist, do early exit
        if not self.overwrite:
            all_exist = True
            for filepath in filepaths:
                if not os.path.isfile(filepath):
                    all_exist = False
                    break
            if all_exist:
                if verbose: print(f"All {split.value} dataset CSV files already exist!")
                return

        # Check that the directory exists and if not, create it
        os.makedirs(os.path.dirname(filepaths[0]), exist_ok=True)
        
        # Create multiprocessing pool
        with mp.Pool(processes=self.num_workers) as pool:
            if verbose: print(f"Generating {split.value} dataset CSV files...")
            
            # Iterate over every chunk of data
            for i in tqdm(range(num_files), position=0, leave=False, disable=not verbose):
                # Skip if filename already exists only if overwrite option is disabled
                if not self.overwrite and os.path.isfile(filepaths[i]):
                    if verbose: print(f"\nCSV File: {filepaths[i]} already exists. Moving on...")
                    continue

                # For each image in the chunk, pre allocate space for data collection
                start = i*self.max_entries_per_file
                end = min((i+1)*self.max_entries_per_file, len(dataset))

                # Obtain the relevant CSV data of all images in this chunk in parallel
                data = list(tqdm(pool.imap(convert_to_array_of_data, zip(range(start, end), [dataset]*(end - start)), self.chunksize), total=end-start, position=1, leave=False, disable=not verbose))

                # File writing
                filepath = filepaths[i]
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

        if verbose: print(f"{split.value} dataset CSV files generated.\n")


def convert_to_array_of_data(arg: Tuple[int, Dataset]):
    """
    Given a dataset sample partitioned by CuPID or SLIC, extract the relevant information to be written
    into a single row in a csv file. This function is expected to be ran in parallel using
    Python multiprocessing.Pool.imap.
    """
    # Obtain partitioned data of the image
    i, dataset = arg
    partitioned_object, label = dataset[i]

    # If label is not an integer (inside an array) extract it
    if type(label) != int:
        label = np.array(label).item()

    # Get csv data
    new_row_entry = partitioned_object.transform_to_csv_data()

    # Add in index and label information
    new_row_entry[0] = i
    new_row_entry[1] = label
    return new_row_entry


"""
Script to create the CSV files.
"""
import time
def main():
    creator = CSV_Dataset_Writer("data/csv/", MyMedMNIST, 16, Partition.CuPID, 
                                 chunksize=50, overwrite=False, num_workers=28)
    start = time.perf_counter()
    creator.create_csv_files(verbose=True)
    end = time.perf_counter()
    return end - start

if __name__ == '__main__':
    time_taken = main()
    print(f"Time taken: {time_taken/60} minutes")
