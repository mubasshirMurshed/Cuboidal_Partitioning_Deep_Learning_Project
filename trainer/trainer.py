# Imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
from typing import Union, Dict
from utils.logger import Logger
from utils.utilities import dirManager, count_trainable_parameters, count_untrainable_parameters, getPythonFilePath, prettyprint
import os
from data.datamodules import DataModule
from shutil import copy
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchmetrics import MetricCollection
from torchinfo import summary
import csv

HYPHEN_COUNT = 80

class Trainer():
    """
    Controls the training process by taking in the model and the respective dataloaders with flags controlling
    certain procedures and checks.
    """
    def __init__(self,
            model: nn.Module, data_module: DataModule, num_classes: int,
            loss_fn: nn.Module = None, optimizer: Optimizer = None, scheduler: LRScheduler = None, hparams: Dict = None, 
            save_every_n_epoch: int = None, allow_log: bool = True, print_conf_matrix: bool = True,
            resume_from_ckpt: Union[str, None]  = None, is_graph_model: bool = True, verbose: bool = True,
            save_best: bool = True, log_to_csv: bool = True
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
        self.scheduler = scheduler
        self.save_every_n_epoch = save_every_n_epoch
        self.allow_log = allow_log
        self.num_classes = num_classes
        self.print_conf_matrix = print_conf_matrix
        self.resume_from_ckpt = resume_from_ckpt
        self.is_graph_model = is_graph_model
        self.verbose = verbose
        self.save_best = save_best
        self.log_to_csv = log_to_csv

        # Create and setup dataloaders
        self.data_module = data_module
        self.training_loader = self.data_module.train_dataloader()
        self.validation_loader = self.data_module.val_dataloader()
        self.test_loader = self.data_module.test_dataloader()

        # Get device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create and setup metric trackers
        self.train_accuracy = MulticlassAccuracy(self.num_classes).to(self.device)
        self.val_accuracy = MulticlassAccuracy(self.num_classes).to(self.device)
        self.val_metrics = MetricCollection( {
            "Top 2 Accuracy" : MulticlassAccuracy(self.num_classes, top_k=2),
            "Top 3 Accuracy" : MulticlassAccuracy(self.num_classes, top_k=3),
            "Confusion Matrix" : MulticlassConfusionMatrix(self.num_classes)
        }, compute_groups=False).to(self.device)
        self.test_metrics = MetricCollection( {
            "Top 1 Accuracy" : MulticlassAccuracy(self.num_classes),
            "Top 2 Accuracy" : MulticlassAccuracy(self.num_classes, top_k=2),
            "Top 3 Accuracy" : MulticlassAccuracy(self.num_classes, top_k=3),
            "Confusion Matrix" : MulticlassConfusionMatrix(self.num_classes)
        }, compute_groups=False).to(self.device)


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
            sys.stdout = Logger(self.log_dir, "output.log", self.verbose)

            # Log hyperparameters
            self.logger.add_hparams(hparam_dict=self.hparams, 
                metric_dict={}, run_name=os.path.dirname(os.path.realpath("main.py")) + os.sep + self.log_dir
            )

            # Save model, datamodule and main python files in logging directory
            model_path = getPythonFilePath(self.model)
            dm_path = getPythonFilePath(self.data_module)
            copy(model_path, self.file_dir)
            copy(dm_path, self.file_dir)
            copy("main.py", self.file_dir)

            # Create csv file to contain results over each epoch
            if self.log_to_csv:
                filepath = self.log_dir + "result_log.csv"
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                csv_file = open(filepath, 'w', newline='')
                writer = csv.writer(csv_file)
                writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy"])

        # Check if model is to be loaded
        if self.resume_from_ckpt is not None:
            self.model.load_state_dict(torch.load(self.resume_from_ckpt))
            print('-' * HYPHEN_COUNT)
            print("Model successfully loaded from " + self.resume_from_ckpt)

        # Move model over to device
        self.model = self.model.to(device=self.device)

        # Print out what dataset is being trained on
        print('-' * HYPHEN_COUNT)
        print(f"Dataset:\t\t{self.data_module.dataset_name.upper()}")
        print('-' * HYPHEN_COUNT)

        # Print out partitioning algorithm and segment number
        print(f"Partition:\t\t{'CuPID' if self.data_module.mode == 'CP' else 'SLIC'} - {self.data_module.num_segments}")
        print('-' * HYPHEN_COUNT)

        # Print ablation code
        print(f"Ablation Code:\t\t{self.data_module.train_set.ablation_code}")
        print('-' * HYPHEN_COUNT)

        # Print model name
        print(f"Model:\t\t\t{self.model.__class__.__name__}")
        print('-' * HYPHEN_COUNT)

        # Print out hyperparameters
        print(f"Hyperparameters:")
        for k, v in self.hparams.items():
            print(f"\t{k:15s} :  {v}")
        print('-' * HYPHEN_COUNT)

        # Print out number of parameters of models
        print(f"Model Summary:")
        num_trainable_parameters = count_trainable_parameters(self.model)
        num_untrainable_parameters = count_untrainable_parameters(self.model)
        total_paramaters = num_trainable_parameters + num_untrainable_parameters
        estimated_size_kb = total_paramaters / 250        
        summary(self.model)
        print(f"Estimated size of model\t{estimated_size_kb:6.2f} KB")
        print('-' * HYPHEN_COUNT)
        print('-' * HYPHEN_COUNT)
        
        # Determine training and validating functions   TODO: Make better when it comes to ensemble and CNN robustness
        if not self.is_graph_model:
            train_fn = self.train
            validate_fn = self.validation
        else:
            train_fn = self.train_G
            validate_fn = self.validation_G

        # Initialise best model metric
        if self.save_best:
            best_val_loss = float("inf")

        # Try-Catch block for allowing graceful finish with Keyboard Interrupts
        try:
            # Loop over requested epochs
            for epoch in range(self.max_epochs):
                # Run training cycle
                train_loss = train_fn()

                # Run validation cycle
                val_loss = validate_fn()

                # Update learning rate
                self.scheduler.step()

                # Print metrics to console
                train_acc = self.train_accuracy.compute()
                val_acc = self.val_accuracy.compute()
                print("Epoch: {}/{}, Train Loss: {:.3f}, Train Acc: {:.2%}, Val Loss: {:.3f}, Val Accuracy: {:.2%}".format(epoch+1, self.max_epochs, train_loss, train_acc.item(), val_loss, val_acc.item()))
                print('-' * HYPHEN_COUNT)

                # Reset metrics
                self.train_accuracy.reset()
                self.val_accuracy.reset()

                if self.allow_log:
                    # Tensorboard logging
                    self.logger.add_scalar("train_loss", train_loss, epoch)
                    self.logger.add_scalar("train_acc", train_acc, epoch)
                    self.logger.add_scalar("val_loss", val_loss, epoch)
                    self.logger.add_scalar("val_acc", val_acc, epoch)

                    # Save best checkpoint
                    if self.save_best and val_loss < best_val_loss:
                        torch.save(self.model.state_dict(), self.ckpt_dir + "best.pt")
                        best_val_loss = val_loss

                    # Saving checkpoints
                    if self.save_every_n_epoch != 0 and ((epoch+1) % self.save_every_n_epoch == 0):
                        filename = f"epoch={epoch+1}-val_loss={val_loss:.4f}-val_acc={val_acc:.4f}.pt"
                        torch.save(self.model.state_dict(), self.ckpt_dir + filename)

                    # Keep track in csv log
                    if self.log_to_csv:
                        writer.writerow([epoch+1, train_loss, train_acc.item(), val_loss, val_acc.item()])
            
            print("Finished Training")

        except KeyboardInterrupt:
            print("Training halted")
        
        # Run a single validation run for confusion matrix
        avg_val_loss = validate_fn(final_metrics=True)
        val_metrics = self.val_metrics.compute()
        self.val_metrics.reset()
        val_acc = self.val_accuracy.compute()
        self.val_accuracy.reset()
        print('-' * HYPHEN_COUNT)
        print("Validation Dataset Results:")
        print('-' * HYPHEN_COUNT)
        print(f"Loss: {avg_val_loss:.5}")
        print(f"Top 1 Accuracy: {val_acc:.2%}")
        print(f"Top 2 Accuracy: {val_metrics['Top 2 Accuracy']:.2%}")
        print(f"Top 3 Accuracy: {val_metrics['Top 3 Accuracy']:.2%}")
        print('-' * HYPHEN_COUNT)

        # Print confusion matrix
        if self.print_conf_matrix:
            prettyprint(val_metrics["Confusion Matrix"].to("cpu").numpy(), self.num_classes)
            print('-' * HYPHEN_COUNT)

        if self.allow_log:
            # Save current model + output log
            torch.save(self.model.state_dict(), self.ckpt_dir + "last.pt")
            print("Logs saved to {}".format(self.ckpt_dir))

            # Close csv file writer
            if self.log_to_csv:
                csv_file.close()


    def test(self, model_ckpt=None) -> None:
        # Load model is a checkpoint is given
        if model_ckpt is not None:
            self.model.load_state_dict(torch.load(model_ckpt))
        else:
            self.model.load_state_dict(torch.load(self.ckpt_dir + "best.pt"))

        # Move model over to device
        self.model = self.model.to(device=self.device)

        # Run test
        if not self.is_graph_model:
            test_fn = self.test_
        else:
            test_fn = self.test_G
        avg_test_loss = test_fn()

        # Compute metrics and print results
        test_metric_results = self.test_metrics.compute()
        self.test_metrics.reset()
        print('-' * HYPHEN_COUNT)
        print("Test Dataset Results:")
        print('-' * HYPHEN_COUNT)
        print(f"Loss: {avg_test_loss:.5}")
        print(f"Top 1 Accuracy: {test_metric_results['Top 1 Accuracy']:.2%}")
        print(f"Top 2 Accuracy: {test_metric_results['Top 2 Accuracy']:.2%}")
        print(f"Top 3 Accuracy: {test_metric_results['Top 3 Accuracy']:.2%}")
        print('-' * HYPHEN_COUNT)

        # Print confusion matrix
        if self.print_conf_matrix:
            prettyprint(test_metric_results["Confusion Matrix"].to("cpu").numpy(), self.num_classes)
            print('-' * HYPHEN_COUNT)
        

    def train(self):
        """
        Performs one training loop over the training data loader.
        """
        # Put the model in training mode
        self.model.train()
        running_loss = 0
        for (i1, i2, i3, i4), labels in tqdm(self.training_loader, leave=False, disable=not self.verbose):
            # Get batch of images and labels
            # inputs, labels = inputs.to(self.device), labels.to(self.device)
            i1, i2, i3, i4, labels = i1.to(self.device), i2.to(self.device), i3.to(self.device), i4.to(self.device), labels.to(self.device)

            # Forward                                         
            self.optimizer.zero_grad()
            outputs = self.model(i1, i2, i3, i4)

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
        self.model.eval() # puts the model in evaluation mode
        running_loss = 0
        
        with torch.no_grad(): # save memory by not saving gradients which we don't need 
            for (i1, i2, i3, i4), labels in tqdm(self.validation_loader, leave=False, disable=not self.verbose):
                # Get batch of images and labels
                # images, labels = images.to(self.device), labels.to(self.device) # put the data on the GPU
                i1, i2, i3, i4, labels = i1.to(self.device), i2.to(self.device), i3.to(self.device), i4.to(self.device), labels.to(self.device)
                
                # Forward
                outputs = self.model(i1, i2, i3, i4)

                # Calculate metrics
                val_loss = self.loss_fn(outputs, labels)
                running_loss += val_loss.item()
        
                # Update trackers
                self.val_accuracy.update(outputs, labels)
                if final_metrics:
                    self.val_metrics.update(outputs, labels)

            return running_loss/len(self.validation_loader)


    def test_(self):
        """
        Performs one test epoch over the test data loader.
        """
        self.model.eval() # puts the model in evaluation mode
        running_loss = 0
        
        with torch.no_grad(): # save memory by not saving gradients which we don't need 
            for (i1, i2, i3, i4), labels in tqdm(self.test_loader, leave=False, disable=not self.verbose):
                # Get batch of images and labels
                # images, labels = images.to(self.device), labels.to(self.device) # put the data on the GPU
                i1, i2, i3, i4, labels = i1.to(self.device), i2.to(self.device), i3.to(self.device), i4.to(self.device), labels.to(self.device)
                
                # Forward
                outputs = self.model(i1, i2, i3, i4)

                # Calculate metrics
                test_loss = self.loss_fn(outputs, labels)
                running_loss += test_loss.item()
        
                # Update trackers
                self.test_metrics.update(outputs, labels)

            return running_loss/len(self.test_loader)


    def train_G(self):
        """
        Performs one training loop over the training data loader for graph neural network training.
        """
        # Put the model in training mode
        self.model.train()
        running_loss = 0
        for batch in tqdm(self.training_loader, leave=False, disable=not self.verbose):
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
        self.model.eval() # puts the model in evaluation mode
        running_loss = 0
        
        with torch.no_grad(): # save memory by not saving gradients which we don't need 
            for batch in tqdm(self.validation_loader, leave=False, disable=not self.verbose):
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
                    self.val_metrics.update(outputs, batch.y)

            return running_loss/len(self.validation_loader) # return average validation loss


    def test_G(self):
        """
        Tests the model's performance over the test dataset provided using graph neural networks.
        """
        self.model.eval() # puts the model in evaluation mode
        running_loss = 0
        
        with torch.no_grad(): # save memory by not saving gradients which we don't need 
            for batch in tqdm(self.test_loader, leave=False, disable=not self.verbose):
                # Get batch of images and labels
                batch = batch.to(self.device) # put the data on the GPU
                
                # Forward
                outputs = self.model(batch.x, batch.edge_index, batch.batch) # passes image to the model, and gets an ouput which is the class probability prediction

                # Calculate metrics
                test_loss = self.loss_fn(outputs, batch.y) # calculates test_loss from model predictions and true labels
                running_loss += test_loss.item()

                # Update trackers
                self.test_metrics.update(outputs, batch.y)

            return running_loss/len(self.test_loader) # return average test loss