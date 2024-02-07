# Imports
from __future__ import annotations
import numpy as np
from numpy import ndarray
from numpy import int8
from typing import Tuple, NewType
from CuPID import CuPID

class CupidPartition:
    def __init__(self, num_cuboids) -> None:
        self.num_cuboids = num_cuboids

    def __call__(self, sample):
        # Preprocess image for partitioning
        data = np.expand_dims(sample, -1)/255
        c = CuPID(data)
        c.partition(self.num_cuboids)
        return c.cuboids