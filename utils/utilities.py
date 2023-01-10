import torch
from torch import Tensor

def adjacent(c1: Tensor, c2: Tensor) -> bool:
    """
    A function that takes two cuboids with five features each:
        1) x_centre
        2) y_centre
        3) colour
        4) width
        5) height
    and determines whether the two cuboids are adjacent in x-y
    cartesian space.

    Args:
    - c1: Tensor
        - A tensor of shape (5)
    - c2: Tensor
        - A tensor of shape (5) 
    """
    x_dist = abs(c1[0] - c2[0])
    y_dist = abs(c1[1] - c2[1])
    return (x_dist <= (c1[3] + c2[3])/2) and (y_dist <= (c1[4] + c2[4])/2)