import numpy as np
from numpy import ndarray
import pandas as pd
from skimage.segmentation import slic

class SLIC:
    def __init__(self, I: ndarray) -> None:
        # Find key information of the image
        self.dtype = I.dtype
        self.I = I.astype(np.float64)
        self.total_spatial_size = np.prod(I.shape[:len(I.shape)-1])
        self.normaliser = 1
        if np.issubdtype(self.dtype, np.integer):
            self.normaliser = np.iinfo(self.dtype).max


    def partition(self, N: int, verbose: bool=False) -> pd.DataFrame:
        # Validate N
        if N > self.total_spatial_size:
            if verbose: print(f"{N = } is greater than the image size. N has been set to {self.total_spatial_size}.")
            N = self.total_spatial_size
        
        # Perform SLIC and obtain a mask
        self.superpixels = slic(self.I, n_segments=N, slic_zero=True)
        self.num_segments = len(np.unique(self.superpixels))

        # Holds information on each segment
        self.segments = []

        # Iterate over each segment and grab the average colour and fill in corresponding region
        for segment_ID in range(1, self.num_segments + 1):
            # Get a mask for the segment
            mask = self.superpixels == segment_ID
            superpixel_region = self.I[mask]

            # Compute the average color of this segment
            mu = superpixel_region.mean(axis=0)/self.normaliser

            # Compute variance
            sigma2 = superpixel_region.var(axis=0)/(self.normaliser**2)

            # Compute size
            no_of_pixels = np.sum(mask)

            # Compute x, y coordinates and max width and height
            segmentation = np.where(mask)
            x_center = segmentation[1].mean()
            y_center = segmentation[0].mean()

            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            width = x_max - x_min + 1

            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))
            height = y_max - y_min + 1

            self.add_segment(x_center, y_center, width, height, no_of_pixels, mu, sigma2)

        self.segments = pd.DataFrame(self.segments)
        return self.segments


    def add_segment(self, x_center, y_center, width, height, n, mu: ndarray, sigma2: ndarray) -> None:
        id = len(self.segments) + 1
        self.segments.append({"id": id,
                             "x_center": x_center,
                             "y_center": y_center,
                             "width": width,
                             "height": height,
                             "n": n,
                             "mu": mu,
                             "sigma2": sigma2})


    def reconstruct(self) -> ndarray:
        # Create an empty image to store the average colors
        I_r = np.zeros(self.I.shape)

        # Iterate over each segment and grab the average colour and fill in corresponding region
        for i in range(len(self.segments)):
            segment = self.segments.iloc[i]

            # Get segment ID in the mask
            segment_ID = segment["id"]

            # Get a mask for the segment
            mask = self.superpixels == segment_ID

            # Assign this color to the output image
            I_r[mask] = segment["mu"]
        
        # Renormalise and convert back to type
        I_r *= self.normaliser
        if np.issubdtype(self.dtype, np.integer) or self.dtype == bool:
            I_r = np.round(I_r)
        I_r = I_r.astype(self.dtype)
        return I_r
    

    def adjacency_matrix(self) -> ndarray:
        """
        Gets the adjacency matrix
        """
        # Create space
        adj_matrix = np.zeros((self.num_segments, self.num_segments), dtype=np.int64)

        # Assign map
        map = self.superpixels - 1

        # Iterate over each cuboid and add edges
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                if i + 1 < map.shape[0] and map[i, j] != map[i + 1, j]:
                    adj_matrix[map[i, j], map[i + 1, j]] = 1
                    adj_matrix[map[i + 1, j], map[i, j]] = 1
                if j + 1 < map.shape[1] and map[i, j] != map[i, j + 1]:
                    adj_matrix[map[i, j], map[i, j + 1]] = 1
                    adj_matrix[map[i, j + 1], map[i, j]] = 1
                if i + 1 < map.shape[0] and j + 1 < map.shape[1] and map[i, j] != map[i + 1, j + 1]:
                    adj_matrix[map[i, j], map[i + 1, j + 1]] = 1
                    adj_matrix[map[i + 1, j + 1], map[i, j]] = 1
                
        
        return adj_matrix