# Deep Learning Applications of CuPID employing Graph Neural Networks
**Written by Mubasshir Murshed.**
Last updated: 19th May, 2024

This software package has been developed under the Engineering FYP Honours Project at Monash University.

The software provides a generalised framework for training classification models using clustering algorithms to form a graph out of images. It has been used specifically to test CuPID based Graph Neural Networks on classification tasks, to study
any advantages or disadvantages it has. The project has been supported by Associate Professor Mehrtash Harandi, Monash University.

The software can extended quite easily to other datasets as well as other partitioning algorithms that can be translated into a Region Adjacency Graph (RAG).

## What does this repository contain?
This repository contains the collection of files and models used. The dataset being used can be found in the *data* directory.

The environment used to run the repository can be found in ***environment.yml***.

All models developed are contained in the *models* directory along with base model information.

All training logs are logged in the saved directory under *saved/DATAMODULE_NAME/ABLATION/MODEL_NAME/RUN_ID/*.

The contents that get saved are:
- checkpoints at specific epochs
- output logs
- configuration data of the run in main.py
- model used for the run
- tensorboard data
- csv file of results at each epoch

## How to generate/process data?

This framework attaches the CuPID algorithm as a transform to the PyTorch Geometric models. However it goes through an intricate data pre-processing pipeline.
The raw datasets are first processed by CuPID and the results are written into a .csv file. A seperate PyTorch Geometric In-Memory Dataset object will load this .csv data efficiently and provide the graphs
to a graph neural network for training/testing. Before model training, datasets need to be created into .csv files by running the script,

    data/source_to_csv.py

The script can be altered on which dataset it is processing.

## How to use this framework?
To create the conda environment containing all the packages required, run the following command in the terminal, which assumes your device has an appropriate Conda package manager:

    conda env create --file environment.yml

To move into this environment, run the following command in the terminal:

    conda activate --n cupid

Now,

    python main.py

will execute and train the selected model on the selected datamodule. Edit the script to change which model and which dataset, as well as other hyperparameters.

## What metrics are logged?

Metrics are printed to the console for training, validation and test datasets. The metrics displayed are Top 1 to 3 accuracies as well
as confusion matrices.
