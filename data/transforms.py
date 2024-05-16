# Imports
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
from clustering.CuPID import CuPID
from clustering.SLIC import SLIC

class CuPIDPartition:
    def __init__(self, num_cuboids, b=0) -> None:
        self.num_cuboids = num_cuboids
        self.b = b

    def __call__(self, sample):
        # Preprocess image for partitioning
        sample = np.asarray(sample)
        if len(sample.shape) != 3:
            sample = np.expand_dims(sample, -1)
        c = CuPID(sample)
        c.partition(self.num_cuboids, b=self.b)
        return c.cuboids, c
    

class CuPIDTransform:
    def __init__(self, num_cuboids, b=0) -> None:
        self.num_cuboids = num_cuboids
        self.b = b

    def __call__(self, sample):
        sample = np.asarray(sample)
        if len(sample.shape) != 3:
            sample = np.expand_dims(sample, -1)
        c = CuPID(sample)
        c.partition(self.num_cuboids, b=self.b)
        I_r = c.reconstruct()
        return I_r.copy()
    

class SLICPartition:
    def __init__(self, num_segments) -> None:
        self.num_segments = num_segments

    def __call__(self, sample):
        # Preprocess image for partitioning
        sample = np.asarray(sample)
        if len(sample.shape) != 3:
            sample = np.expand_dims(sample, -1)
        s = SLIC(sample)
        s.partition(N=self.num_segments)
        return s.segments, s


class SLICTransform:
    def __init__(self, num_segments) -> None:
        self.num_segments = num_segments

    def __call__(self, sample):
        # Preprocess image for partitioning
        sample = np.asarray(sample)
        if len(sample.shape) != 3:
            sample = np.expand_dims(sample, -1)
        s = SLIC(sample)
        s.partition(self.num_segments)
        I_r = s.reconstruct()
        return I_r.copy()