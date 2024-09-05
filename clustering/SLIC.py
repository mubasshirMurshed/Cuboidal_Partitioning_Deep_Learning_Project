# Imports
import numpy as np
from numpy.typing import NDArray
from numpy import float64
import pandas as pd
from skimage.segmentation import slic


class SLIC:
    """
    Class houses important SLIC partitioning algorithms and encapsulates the information into an easily readable
    pandas DataFrame. Uses the SLICO variant with an initial compactness of 1.
    """
    def __init__(self, I: NDArray) -> None:
        """
        Instantiates a SLIC object of a point cloud datum, I.

        Args:
        - I: NDArray
            - The point cloud data, which can represent a dp-dimensional data with df size feature vector
        """
        # Find key information of the image
        self.dtype = I.dtype
        self.I = I.astype(np.float64)
        self.size = np.prod(I.shape[:len(I.shape)-1])

        # Get normalising factors
        self.normaliser = 1
        if np.issubdtype(self.dtype, np.integer):
            self.normaliser = np.iinfo(self.dtype).max


    def partition(self, N: int, *, compactness: float=1, verbose: bool=False) -> pd.DataFrame:
        """
        Partitions the point cloud datum, I, into approximately N superpixels. Due to how SLIC works,
        the result may be much lower than the requested amount.

        Args:
        - N: int
            - The number of target superpixels to generate
        - compactness: float
            - The compactness factor of the SLIC algorithm, default=1
        - verbose: bool
            - Flag to allow printing of information, default=False

        Return:
        - A pandas DataFrame recording each superpixel ever generated and their properties.
        """
        # Validate N
        if N > self.size:
            if verbose: print(f"{N = } is greater than the image size. N has been set to {self.size}.")
            N = self.size
        
        # Perform SLIC and obtain a mask
        self.superpixel_mask = slic(self.I, n_segments=N, slic_zero=True, compactness=compactness) - 1
        self.num_segments = len(np.unique(self.superpixel_mask))

        # Holds information on each segment
        self.segments = []

        # Iterate over each segment and grab the average colour and fill in corresponding region
        for segment_ID in range(self.num_segments):
            # Get a mask for the segment
            mask = self.superpixel_mask == segment_ID
            superpixel = self.I[mask]

            # Compute the average color of this segment
            mu = superpixel.mean(axis=0)/self.normaliser

            # Compute variance of superpixel
            sigma2 = superpixel.var(axis=0)/(self.normaliser**2)

            # Compute size
            no_of_pixels = np.sum(mask)

            # Compute x, y coordinates and max width and height
            superpixel_idxs = np.where(mask)
            x_center = superpixel_idxs[1].mean()
            y_center = superpixel_idxs[0].mean()

            x_min = int(np.min(superpixel_idxs[1]))
            x_max = int(np.max(superpixel_idxs[1]))
            width = x_max - x_min + 1

            y_min = int(np.min(superpixel_idxs[0]))
            y_max = int(np.max(superpixel_idxs[0]))
            height = y_max - y_min + 1

            # Add to records
            self._add_segment(x_center, y_center, width, height, no_of_pixels, mu, sigma2)

        # Convert records to a DataFrame
        self.segments = pd.DataFrame(self.segments)
        return self.segments


    def _add_segment(self, x_center: float, y_center: float, width: int, height: int, n: int, mu: NDArray[float64], sigma2: NDArray[float64]) -> None:
        """
        Adds superpixel segment to the records.
        """
        # Get ID of segment to be added
        id = len(self.segments)
        self.segments.append({"id": id,
                             "x_center": x_center,
                             "y_center": y_center,
                             "width": width,
                             "height": height,
                             "n": n,
                             "mu": mu,
                             "sigma2": sigma2})


    def reconstruct(self) -> NDArray:
        """
        Obtains superpixels of the partition and uses them to reconstruct the partitioned image.
        """
        # Create an empty image to store the average colors
        I_r = np.zeros(self.I.shape)

        # Iterate over each segment and grab the average colour and fill in corresponding region
        for i in range(len(self.segments)):
            segment = self.segments.iloc[i]

            # Get segment ID in the mask
            segment_ID = segment["id"]

            # Get a mask for the segment
            mask = self.superpixel_mask == segment_ID

            # Assign this color to the output image
            I_r[mask] = segment["mu"]
        
        # Renormalise and restore back to original data type
        I_r *= self.normaliser
        if np.issubdtype(self.dtype, np.integer) or self.dtype == bool:
            I_r = np.round(I_r)
        I_r = I_r.astype(self.dtype)
        return I_r
    

    def adjacency_matrix(self) -> NDArray[np.int64]:
        """
        Gets the adjacency matrix based on representing superpixels as vertices and edges as adjacent superpixels
        defined by "touching" each other in an 8-directional semantic.
        """
        # Create space
        adj_matrix = np.zeros((self.num_segments, self.num_segments), dtype=np.int64)

        # Assign map
        map = self.superpixel_mask

        # Iterate over each cuboid and add edges
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                # Checks bottom
                if i + 1 < map.shape[0] and map[i, j] != map[i + 1, j]:
                    adj_matrix[map[i, j], map[i + 1, j]] = 1
                    adj_matrix[map[i + 1, j], map[i, j]] = 1
                
                # Checks right
                if j + 1 < map.shape[1] and map[i, j] != map[i, j + 1]:
                    adj_matrix[map[i, j], map[i, j + 1]] = 1
                    adj_matrix[map[i, j + 1], map[i, j]] = 1

                # Checks bottom right
                if i + 1 < map.shape[0] and j + 1 < map.shape[1] and map[i, j] != map[i + 1, j + 1]:
                    adj_matrix[map[i, j], map[i + 1, j + 1]] = 1
                    adj_matrix[map[i + 1, j + 1], map[i, j]] = 1

                # Checks bottom left
                if i + 1 < map.shape[0] and j > 0 and map[i, j] != map[i + 1, j - 1]:
                    adj_matrix[map[i, j], map[i + 1, j - 1]] = 1
                    adj_matrix[map[i + 1, j - 1], map[i, j]] = 1

        return adj_matrix
    
    
    def transform_to_csv_data(self):       
        # Initialise first few column values
        table = self.segments
        shape = self.I.shape
        no_of_nodes = len(table)
        no_of_features = shape[-1]
        new_row_entry = [None, None, 0, no_of_nodes, no_of_features, shape[0], shape[1]]

        # Add in center x coordinates segment
        for _, row in table.iterrows():
            new_row_entry.append(round(row["x_center"], 2))

        # Add in center y coordinates for each segment
        for _, row in table.iterrows():
            new_row_entry.append(round(row["y_center"], 2))

        # Add in colour values for each segment (restore float to be between 0-255)
        for j in range(no_of_features):
            for _, row in table.iterrows():
                new_row_entry.append(np.around(row["mu"][j]*255, 2))
        
        # Add in num of pixels for each segment
        for _, row in table.iterrows():
            new_row_entry.append(row["n"])

        # Add in box angles for each segment
        for _, row in table.iterrows():
            new_row_entry.append(round(np.arctan(row["height"]/row["width"])*180/np.pi, 2))        # This is hardcoded to 2D

        # Add in box width for each segment
        for _, row in table.iterrows():
            new_row_entry.append(row["width"])

        # Add in box height for each segment
        for _, row in table.iterrows():
            new_row_entry.append(row["height"])

        # Add in standard deviation for each segment w.r.t original sections (scaled to 255 colour space)
        for j in range(no_of_features):
            for _, row in table.iterrows():
                new_row_entry.append(np.around(np.sqrt(row["sigma2"][j])*255, 2))

        # Calculate number of edges present
        adj_matrix = self.adjacency_matrix()
        no_of_edges = np.sum(adj_matrix)
        coo_src, coo_dst = np.where(adj_matrix)

        # Add the edge information
        new_row_entry.append(no_of_edges)
        new_row_entry.extend(coo_src)
        new_row_entry.extend(coo_dst)

        return new_row_entry