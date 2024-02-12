# Imports
from __future__ import annotations
import numpy as np
from datasets.cupid import CuPID

class CupidPartition:
    def __init__(self, num_cuboids, b=0) -> None:
        self.num_cuboids = num_cuboids
        self.b = b

    def __call__(self, sample):
        # Preprocess image for partitioning
        data = np.expand_dims(sample, -1)
        c = CuPID(data)
        c.partition(self.num_cuboids, b=self.b)
        return c.cuboids