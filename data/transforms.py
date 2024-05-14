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
        return c.cuboids
    

class SLICPartition:
    def __init__(self, num_segments) -> None:
        pass

    def __call__(self, sample):
        # Preprocess image for partitioning
        sample = np.asarray(sample)
        if len(sample.shape) != 3:
            sample = np.expand_dims(sample, -1)
        c = SLIC(sample)
        pass