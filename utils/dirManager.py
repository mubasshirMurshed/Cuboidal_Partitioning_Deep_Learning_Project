import torch.nn as nn
from datetime import datetime
from os import makedirs
from dataModules.dataModule import DataModule

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

    # Get current time
    d = datetime.now()
    dateString = f"{d.year:4d}-{d.month:02d}-{d.day:02d}__{d.hour:02d}-{d.minute:02d}-{d.second:02d}"

    # Create Run ID and directory paths
    runID =  'Run_ID__' + dateString
    log_dir = f"saved/{dataModuleName}/{modelName}/{runID}"
    ckpt_dir = log_dir + "/checkpoints"

    # Create directories if they do not exist
    makedirs(ckpt_dir)
    return log_dir, ckpt_dir