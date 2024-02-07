# Imports
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typing import List, Union, Dict, Callable
import sys
from utils.logger import Logger
from utils.utilities import dirManager, count_trainable_parameters, count_untrainable_parameters, getPythonFilePath
import os
from dataModules.dataModule import DataModule
from shutil import copy
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchmetrics import MetricCollection


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
        self.training_loader = self.data_module.train_dataloader()
        self.validation_loader = self.data_module.val_dataloader()

        # Get device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create and setup metric trackers
        self.train_accuracy = MulticlassAccuracy(self.num_classes).to(self.device)
        self.val_accuracy = MulticlassAccuracy(self.num_classes).to(self.device)
        self.final_metrics = MetricCollection( {
            "Top 2 Accuracy" : MulticlassAccuracy(self.num_classes, top_k=2),
            "Top 3 Accuracy" : MulticlassAccuracy(self.num_classes, top_k=3),
            "Confusion Matrix" : MulticlassConfusionMatrix(self.num_classes)
        }).to(self.device)
        
        print('-' * 80)


    def fit(self):
        """
        Fits dataset onto model.
        """
        # Get directory information
        if self.allow_log:
            # Create the directories required
            self.log_dir, self.ckpt_dir, self.file_dir = dirManager(self.model, self.data_module)

            # Set tensorboard writer
            self.logger = SummaryWriter(self.log_dir)

            # Set stdout logger
            sys.stdout = Logger(self.log_dir, "/output.log")

            # Log hyperparameters
            self.logger.add_hparams(hparam_dict=self.hparams, 
                metric_dict={}, run_name=os.path.dirname(os.path.realpath("main.py")) + os.sep + self.log_dir
            )

            # Save model, datamodule and main python files in log_dir
            model_path = getPythonFilePath(self.model)
            dm_path = getPythonFilePath(self.data_module)
            copy(model_path, self.file_dir)
            copy(dm_path, self.file_dir)
            copy("main.py", self.file_dir)

        # Check if model is to be loaded
        if self.resume_from_ckpt is not None:
            self.model.load_state_dict(torch.load(self.resume_from_ckpt))
            print("Model successfully loaded from " + self.resume_from_ckpt)
            print('-' * 80)

        # Move model over to device
        self.model = self.model.to(device=self.device)

        # Print out what data module is being trained on
        print(f"Data Module:\t\t{self.data_module.__class__.__name__}")
        print('-' * 80)

        # Print ablation code
        print(f"Ablation Code:\t\t{self.data_module.train_set.ablation_code}")
        print('-' * 80)

        # Print model name
        print(f"Model:\t\t\t{self.model.__class__.__name__}")
        print('-' * 80)

        # Print out hyperparameters
        print(f"Hyperparameters:")
        for k, v in self.hparams.items():
            print(f"\t{k:15s} :  {v}")
        print('-' * 80)

        # Print out number of parameters of models
        print(f"Model Summary:")
        num_trainable_parameters = count_trainable_parameters(self.model)
        num_untrainable_parameters = count_untrainable_parameters(self.model)
        total_paramaters = num_trainable_parameters + num_untrainable_parameters
        estimated_size_kb = total_paramaters / 250
        model_summary = f"\
\t{num_trainable_parameters/1000:6.2f} K \t Trainable params\n\
\t{num_untrainable_parameters/1000:6.2f} K \t Non-trainable params\n\
\t{total_paramaters/1000:6.2f} K \t Total params\n\
\t{estimated_size_kb:6.2f} KB\t Estimated size of model"
        print(model_summary)
        print('-' * 80)
        
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
                val_loss = validate_fn()

                # Print metrics to console
                train_acc = self.train_accuracy.compute()
                val_acc = self.val_accuracy.compute()
                print("Epoch: {}/{}, Train Loss: {:.3f}, Train Acc: {:.2%}, Val Loss: {:.3f}, Val Accuracy: {:.2%}".format(epoch+1, self.max_epochs, train_loss, train_acc.item(), val_loss, val_acc.item()))
                print('-' * 80)

                # Reset metrics
                self.train_accuracy.reset()
                self.val_accuracy.reset()

                if self.allow_log:
                    # Tensorboard logging
                    self.logger.add_scalar("train_loss", train_loss, epoch)
                    self.logger.add_scalar("train_acc", train_acc, epoch)
                    self.logger.add_scalar("val_loss", val_loss, epoch)
                    self.logger.add_scalar("val_acc", val_acc, epoch)

                    # Saving checkpoints
                    if ((epoch+1) % self.save_every_n_epoch == 0):
                        filename = f"/epoch={epoch+1}-val_loss={val_loss:.4f}-val_acc={val_acc:.4f}.pt"
                        torch.save(self.model.state_dict(), self.ckpt_dir + filename)
            
            print("Finished Training")

        except KeyboardInterrupt:
            print("Training halted")
        
        # Print confusion matrix
        if self.print_conf_matrix:
            validate_fn(final_metrics=True)
            extra_metrics = self.final_metrics.compute()
            self.final_metrics.reset()
            val_acc = self.val_accuracy.compute()
            self.val_accuracy.reset()
            print('-' * 80)
            print(f"Top 1 Accuracy: {val_acc:.2%}")
            print(f"Top 2 Accuracy: {extra_metrics['Top 2 Accuracy']:.2%}")
            print(f"Top 3 Accuracy: {extra_metrics['Top 3 Accuracy']:.2%}")
            print('-' * 80)
            print(extra_metrics["Confusion Matrix"].to("cpu").numpy())
            print('-' * 80)

        if self.allow_log:
            # Save current model + output log
            torch.save(self.model.state_dict(), self.ckpt_dir + "/last.pt")
            print("Logs saved to {}".format(self.ckpt_dir))

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
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward                                         
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Backward
            loss = self.loss_fn(outputs, labels)
            running_loss += loss.item()
            loss.backward()

            # Update accuracies
            self.train_accuracy.update(outputs, labels)
            
            # Update weights
            self.optimizer.step()

        return running_loss/len(self.training_loader)


    def validation(self, final_metrics: bool=False):
        """
        Performs one validation loop over the validation data loader.
        """
        self.model.eval() # puts the model in validation mode
        running_loss = 0
        
        with torch.no_grad(): # save memory by not saving gradients which we don't need 
            for images, labels in tqdm(self.validation_loader, leave=False):
                # Get batch of images and labels
                images, labels = images.to(self.device), labels.to(self.device) # put the data on the GPU
                
                # Forward
                outputs = self.model(images)

                # Calculate metrics
                val_loss = self.loss_fn(outputs, labels)
                running_loss += val_loss.item()
        
                # Update trackers
                self.val_accuracy.update(outputs, labels)
                if final_metrics:
                    self.final_metrics.update(outputs, labels)

            return running_loss/len(self.validation_loader)


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
            running_loss += loss.item() # sum total loss in current epoch for print later
            loss.backward() # calculates gradient w.r.t to loss for all parameters in model that have requires_grad=True

            # Update accuracies
            self.train_accuracy.update(outputs, batch.y)
            
            # Update weights
            self.optimizer.step() # iterate over all parameters in the model with requires_grad=True and update their weights.

        return running_loss/len(self.training_loader) # returns the average training loss for the epoch


    def validation_G(self, final_metrics: bool=False):
        """
        Performs one validation loop over the validation data loader for graph neural network training.
        """
        self.model.eval() # puts the model in validation mode
        running_loss = 0
        
        with torch.no_grad(): # save memory by not saving gradients which we don't need 
            for batch in tqdm(self.validation_loader, leave=False):
                # Get batch of images and labels
                batch = batch.to(self.device) # put the data on the GPU
                
                # Forward
                outputs = self.model(batch.x, batch.edge_index, batch.batch) # passes image to the model, and gets an ouput which is the class probability prediction

                # Calculate metrics
                val_loss = self.loss_fn(outputs, batch.y) # calculates val_loss from model predictions and true labels
                running_loss += val_loss.item()

                # Update trackers
                self.val_accuracy.update(outputs, batch.y)
                if final_metrics:
                    self.final_metrics.update(outputs, batch.y)

            return running_loss/len(self.validation_loader) # return average validation loss