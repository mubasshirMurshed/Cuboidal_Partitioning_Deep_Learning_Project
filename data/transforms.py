import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
from clustering.CuPID import CuPID
from clustering.SLIC import SLIC


class CuPIDPartition:
    """
    A transform that partitions an image using CuPID and returns the CuPID object after
    partitioning has been done. Only suitable for 2D spatial data. Data in the shape of
    (H x W) will have an extra channel of 1 added on to make it (H x W x 1).
    """
    def __init__(self, num_cuboids, b=0) -> None:
        
        self.num_cuboids = num_cuboids
        self.b = b

    def __call__(self, sample):
        """
        Given a sample, partitions using CuPID.
        """
        # Preprocess image for partitioning by converting to numpy
        sample = np.asarray(sample)

        # If grayscale, then expand to make it three dimensions
        if len(sample.shape) != 3:
            sample = np.expand_dims(sample, -1)

        # Perform CuPID
        c = CuPID(sample)
        c.partition(self.num_cuboids, b=self.b)
        return c


class CuPIDTransform:
    """
    A transform that partitions an image using CuPID and returns the reconstruction.
    Only suitable for 2D spatial data. Data in the shape of (H x W) will have an extra
    channel of 1 added on to make it (H x W x 1).
    """
    def __init__(self, num_cuboids, b=0) -> None:
        self.num_cuboids = num_cuboids
        self.b = b

    def __call__(self, sample):
        """
        Given a sample, partitions using CuPID.
        """
        # Preprocess image for partitioning by converting to numpy
        sample = np.asarray(sample)

        # If grayscale, then expand to make it three dimensions
        if len(sample.shape) != 3:
            sample = np.expand_dims(sample, -1)
        
        # Perform CuPID
        c = CuPID(sample)
        c.partition(self.num_cuboids, b=self.b)

        # Reconstruct image
        I_r = c.reconstruct()
        return I_r.copy()


class SLICPartition:
    """
    A transform that partitions an image using SLIC and returns the SLIC object after
    partitioning has been done. Only suitable for 2D spatial data. Data in the shape of
    (H x W) will have an extra channel of 1 added on to make it (H x W x 1).
    """
    def __init__(self, num_segments, compactness=1) -> None:
        self.num_segments = num_segments
        self.compactness = compactness

    def __call__(self, sample):
        """
        Given a sample, partitions using SLIC.
        """
        # Preprocess image for partitioning by converting to numpy
        sample = np.asarray(sample)

        # If grayscale, then expand to make it three dimensions
        if len(sample.shape) != 3:
            sample = np.expand_dims(sample, -1)

        # Perform SLIC
        s = SLIC(sample)
        s.partition(N=self.num_segments, compactness=self.compactness)
        return s


class SLICTransform:
    """
    A transform that partitions an image using SLIC and returns the reconstruction. Only suitable 
    for 2D spatial data. Data in the shape of (H x W) will have an extra channel of 1 added on to 
    make it (H x W x 1).
    """
    def __init__(self, num_segments, compactness=1) -> None:
        self.num_segments = num_segments
        self.compactness = compactness

    def __call__(self, sample):
        """
        Given a sample, partitions using SLIC.
        """
        # Preprocess image for partitioning by converting to numpy
        sample = np.asarray(sample)

        # If grayscale, then expand to make it three dimensions
        if len(sample.shape) != 3:
            sample = np.expand_dims(sample, -1)

        # Perform SLIC
        s = SLIC(sample)
        s.partition(self.num_segments, compactness=self.compactness)

        # Reconstruct image
        I_r = s.reconstruct()
        return I_r.copy()
