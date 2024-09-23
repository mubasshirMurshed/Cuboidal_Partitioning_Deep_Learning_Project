from enum import Enum

class Partition(Enum):
    CuPID = "CP"
    SLIC = "SP"
    CuPID45 = "CP45"


class Split(Enum):
    TRAIN = "Train"
    VALIDATION = "Validation"
    TEST = "Test"
