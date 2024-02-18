import torch.nn as nn
from datetime import datetime
from os import makedirs
from dataModules.dataModule import DataModule
from typing import Callable


def dirManager(model: nn.Module, data_module: DataModule):
    """
    Use model information to create the specific directories for logging and checkpoints
    be stored in.

    Args:
    - model: nn.Module
        - The model being trained
    """
    # Get name of the model and data module class
    modelName = model._get_name()
    dataModuleName = data_module.__class__.__name__
    ablationCode = getattr(data_module.train_set, "ablation_code", None)

    # Get current time
    d = datetime.now()
    dateString = f"{d.year:4d}-{d.month:02d}-{d.day:02d}__{d.hour:02d}-{d.minute:02d}-{d.second:02d}"

    # Create Run ID and directory paths
    runID =  'Run_ID__' + dateString
    if ablationCode is not None:
        log_dir = f"saved/{dataModuleName}/{ablationCode}/{modelName}/{runID}"
    else:
        log_dir = f"saved/{dataModuleName}/{modelName}/{runID}"
    ckpt_dir = log_dir + "/checkpoints"
    file_dir = log_dir + "/python_files"

    # Create directories if they do not exist
    makedirs(ckpt_dir)
    makedirs(file_dir)
    return log_dir, ckpt_dir, file_dir


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_untrainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def getPythonFilePath(obj):
    return obj.__module__.replace(".", "/") + ".py"

def generateConfusionMatrix(self, validate_fn: Callable):
        """
        Generates a confusion matrix on the current model perfomance
        """
        # Create n x n empty matrix where dataset has n categories
        matrix = [0]*self.num_classes
        for i in range(self.num_classes):
            matrix[i] = [0]*self.num_classes

        # Fill in confusion matrix
        validate_fn(conMatrix=matrix)

        # Print formatted table heading
        for i in range(self.num_classes):
            print(f"{i:6d}", end="")
        print("")

        # Print table body
        for i, row in enumerate(matrix):
            print(f"{i}", end="")
            categoryTotal = sum(row)
            for elem in row:
                print(f"{elem*100/categoryTotal:5.0f} ", end="")
            print(f"{sum(row):7d}")