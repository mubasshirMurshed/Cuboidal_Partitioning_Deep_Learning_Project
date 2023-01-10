# Imports
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import List, Union, Dict
import sys
from utils.logger import Logger
from utils.dirManager import dirManager
import os
from dataModules.dataModule import DataModule


class Trainer():
    """
    Controls the training process by taking in the model and the respective dataloaders with flags controlling
    certain procedures and checks.
    """
    def __init__(self,
            model: nn.Module, data_module: DataModule, 
            loss_fn: nn.Module, optimizer: Optimizer, hparams: Dict, 
            save_every_n_epoch: int, allow_log: bool, num_classes: int, print_conf_matrix: bool = True,
            resume_from_ckpt: Union[str, None]  = None, is_graph_model: bool = False
        ) -> None:
        """
        Saves model, datasets/loaders and all flags passed.

        Args:
        - model: nn.Module
            - Model being trained
        - data_module: DataModule
            - A data module that houses the dataset and data loaders for training and validation
        - loss_fn: Module
            - Function used to calculate loss
        - optimizer: Optimizer
            - Optimizer algorithm used to update gradients
        - hparams: Dict
            - Dictionary of hyperparameters of the run
        - save_every_n_epoch: int
            - Flag to determine at what rate model checkpoints should be saved
        - allow_log: bool
            - Flag to determine whether to allow logging or not
        - num_classes: int
            - Number of classes in dataset
        - print_conf_matrix: bool
            - Flag for deciding to print a confusion matrix or not
        - resume_from_ckpt: str | None
            - A filepath to a checkpoint to resume training from under new given hyperparameters,
              if None, training will begin from scratch
        - is_graph_model: bool
            - A boolean value on whether the model is a GNN or not
        """
        # Save attributes
        self.model = model
        self.hparams = hparams
        self.max_epochs = self.hparams["max_epochs"]
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_every_n_epoch = save_every_n_epoch
        self.allow_log = allow_log
        self.num_classes = num_classes
        self.print_conf_matrix = print_conf_matrix
        self.resume_from_ckpt = resume_from_ckpt
        self.is_graph_model = is_graph_model

        # Create and setup dataloaders
        self.data_module = data_module
        self.data_module.setup()
        self.training_loader = self.data_module.train_dataloader()
        self.validation_loader = self.data_module.val_dataloader()
        
        # Get directory information
        self.log_dir, self.ckpt_dir = dirManager(model, data_module)
        self.logger = SummaryWriter(self.log_dir)

        # Get device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Check if model is to be loaded
        if resume_from_ckpt is not None:
            model.load_state_dict(torch.load(resume_from_ckpt))
            print("Model successfully loaded from " + resume_from_ckpt)
            print('-' * 20)


    def fit(self):
        """
        Fits dataset onto model.
        """
        # Set up logging
        if self.allow_log:
            # Set stdout logger
            sys.stdout = Logger(self.log_dir, "/output.log")

            # Log hyperparameters
            self.logger.add_hparams(hparam_dict=self.hparams, 
                metric_dict={}, run_name=os.path.dirname(os.path.realpath("main.py")) + os.sep + self.log_dir
            )

        # Move model over to device
        self.model.to(device=self.device)

        # Print out hyperparameters
        print("Hyperparamaters")
        print(self.hparams)
        print('-' * 20)
        
        # Determine training and validating functions
        if not self.is_graph_model:
            train_fn = self.train
            validate_fn = self.validation
        else:
            train_fn = self.train_G
            validate_fn = self.validation_G

        # Try-Catch block for allowing graceful finish with Keyboard Interrupts
        try:
            # Loop over requested epochs
            for epoch in range(self.max_epochs):
                # Run training cycle
                train_loss = train_fn()

                # Run validation cycle
                val_loss, accuracy = validate_fn()

                # Print metrics to console
                print("Epoch: {}/{}, Training Loss: {:.3f}, Val Loss: {:.3f}, Val Accuracy: {:.2f}%".format(epoch+1, self.max_epochs, train_loss, val_loss, accuracy*100))
                print('-' * 20)

                if self.allow_log:
                    # Tensorboard logging
                    self.logger.add_scalar("train_loss", train_loss, epoch)
                    self.logger.add_scalar("val_loss", val_loss, epoch)
                    self.logger.add_scalar("val_acc", accuracy, epoch)

                    # Saving checkpoints
                    if ((epoch+1) % self.save_every_n_epoch == 0):
                        filename = f"/epoch={epoch+1}-val_loss={val_loss:.4f}-val_acc={accuracy:.4f}.pt"
                        torch.save(self.model.state_dict(), self.ckpt_dir + filename)
            
            print("Finished Training")

        except KeyboardInterrupt:
            print("Training halted")
        
        # Print confusion matrix
        if self.print_conf_matrix:
            self.generateConfusionMatrix(validate_fn=validate_fn)
        
        if self.allow_log:
            # Save current model + output log
            torch.save(self.model.state_dict(), self.ckpt_dir + "/last.pt")

            # Close logging
            self.logger.close()


    def train(self):
        """
        Performs one training loop over the training data loader.
        """
        # Put the model in training mode
        self.model.train()
        running_loss = 0
        for i, data in enumerate(tqdm(self.training_loader, leave=False), 0):
            # Get batch of images and labels
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device) # puts the data on the GPU

            # Forward                                         
            self.optimizer.zero_grad() # clear the gradients in model parameters
            outputs = self.model(inputs) # forward pass and get predictions

            # Backward
            loss = self.loss_fn(outputs, labels) # calculate loss
            loss.backward() # calculates gradient w.r.t to loss for all parameters in model that have requires_grad=True
            
            # Update weights
            self.optimizer.step() # iterate over all parameters in the model with requires_grad=True and update their weights.

            running_loss += loss.item() # sum total loss in current epoch for print later

        return running_loss/len(self.training_loader) # returns the average training loss for the epoch


    def validation(self, conMatrix: Union[List[List[int]], None] = None):
        """
        Performs one validation loop over the validation data loader.
        """
        self.model.eval() # puts the model in validation mode
        running_loss = 0
        total = 0
        correct = 0
        
        with torch.no_grad(): # save memory by not saving gradients which we don't need 
            for images, labels in tqdm(self.validation_loader, leave=False):
                # Get batch of images and labels
                images, labels = images.to(self.device), labels.to(self.device) # put the data on the GPU
                
                # Forward
                outputs = self.model(images) # passes image to the model, and gets an ouput which is the class probability prediction

                # Calculate metrics
                val_loss = self.loss_fn(outputs, labels) # calculates val_loss from model predictions and true labels
                running_loss += val_loss.item()
                _, predicted = torch.max(outputs, 1) # turns class probability predictions to class labels
                total += labels.size(0) # sums the number of predictions
                correct += (predicted == labels).sum().item() # sums the number of correct predictions
        
                # Update confusion matrix
                if conMatrix is not None:
                    for i in range(len(labels)):
                        conMatrix[labels[i]][predicted[i]] += 1

            return running_loss/len(self.validation_loader), correct/total # return average validation loss, accuracy


    def generateConfusionMatrix(self, validate_fn):
        """
        Generates a confusion matrix on the current model perfomance
        """
        # Create n x n empty matrix where dataset has n categories
        matrix = [0]*self.num_classes
        for i in range(self.num_classes):
            matrix[i] = [0]*self.num_classes

        # Fill in confusion matrix
        self.validate_fn(conMatrix=matrix)

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


    def train_G(self):
        """
        Performs one training loop over the training data loader for graph neural network training.
        """
        # Put the model in training mode
        self.model.train()
        running_loss = 0
        for batch in tqdm(self.training_loader, leave=False):
            # Get batch of images and labels
            batch = batch.to(self.device) # puts the data on the GPU

            # Forward                                         
            self.optimizer.zero_grad() # clear the gradients in model parameters
            outputs = self.model(batch.x.float(), batch.edge_index, batch.batch) # forward pass and get predictions

            # Backward
            loss = self.loss_fn(outputs, batch.y) # calculate loss
            loss.backward() # calculates gradient w.r.t to loss for all parameters in model that have requires_grad=True
            
            # Update weights
            self.optimizer.step() # iterate over all parameters in the model with requires_grad=True and update their weights.

            running_loss += loss.item() # sum total loss in current epoch for print later

        return running_loss/len(self.training_loader) # returns the average training loss for the epoch


    def validation_G(self, conMatrix: Union[List[List[int]], None] = None):
        """
        Performs one validation loop over the validation data loader for graph neural network training.
        """
        self.model.eval() # puts the model in validation mode
        running_loss = 0
        total = 0
        correct = 0
        
        with torch.no_grad(): # save memory by not saving gradients which we don't need 
            for batch in tqdm(self.validation_loader, leave=False):
                # Get batch of images and labels
                batch = batch.to(self.device) # put the data on the GPU
                
                # Forward
                outputs = self.model(batch.x.float(), batch.edge_index, batch.batch) # passes image to the model, and gets an ouput which is the class probability prediction

                # Calculate metrics
                val_loss = self.loss_fn(outputs, batch.y) # calculates val_loss from model predictions and true labels
                running_loss += val_loss.item()
                _, predicted = torch.max(outputs, 1) # turns class probability predictions to class labels
                total += batch.y.size(0) # sums the number of predictions
                correct += (predicted == batch.y).sum().item() # sums the number of correct predictions

                # Update confusion matrix
                if conMatrix is not None:
                    for i in range(len(batch.y)):
                        conMatrix[batch.y[i]][predicted[i]] += 1

            return running_loss/len(self.validation_loader), correct/total # return average validation loss, accuracy