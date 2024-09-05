from enum import Enum

class Partition(Enum):
    CuPID = "CP"
    SLIC = "SP"


class Split(Enum):
    TRAIN = "Train"
    VALIDATION = "Validation"
    TEST = "Test"
