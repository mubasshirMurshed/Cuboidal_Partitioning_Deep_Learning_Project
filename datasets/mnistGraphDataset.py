# Imports
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
import pandas as pd
from tqdm import tqdm
from utils.utilities import adjacent

class MNISTCuboidalGraphDataset(Dataset):
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