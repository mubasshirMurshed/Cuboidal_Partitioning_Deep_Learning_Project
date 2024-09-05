# Imports
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import torch
from torch import nn
from data.datamodules import Graph_DataModule_CSV, Graph_DataModule
from models.GAT_Modelv2 import GAT_Modelv2
from data.data_classes import MyMNIST, MyCIFAR_10, MyMedMNIST, MyOmniglot
from tools import Trainer
import torch_geometric
from enums import Partition
# torch.backends.cudnn.deterministic=True
# torch.use_deterministic_algorithms(mode=True)

def main():
    # Seeding
    seed = 42
    torch_geometric.seed_everything(seed)

    # Define hyperparameters
    hparams = {
        "max_epochs" : 100,
        "learning_rate" : 0.001,
        "batch_size" : 64,
        "scheduler_step": 20,
        "scheduler_decay" : 0.8,
        "weight_decay" : 0.01
    }

    # Create data module
    features = {"x_center":True, "y_center":True, "colour":True, "width":True, "height":True}
    data_module = Graph_DataModule_CSV(
        dataset=MyMNIST,
        num_segments=64,
        batch_size=hparams["batch_size"],
        mode=Partition.CuPID,
        num_workers=1,
        features=features
    )

    # Instantiate model
    num_classes = 10                # <------- CHANGE THIS BETWEEN DATASETS!!!!!!!!!!!!
    model = GAT_Modelv2(num_features=data_module.train_set.num_features, num_classes=num_classes)

    # Initialise loss function, optimizer and LR scheduler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.5, 5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, hparams["scheduler_step"], hparams["scheduler_decay"])

    # Add options to hyperparameters
    hparams["optimizer"] = optimizer.__class__.__name__
    hparams["loss_fn"] = loss_fn.__class__.__name__
    hparams["scheduler"] = scheduler.__class__.__name__

    # Define flags
    allow_log = False
    save_every_n_epoch = 1
    save_top_k = 10
    resume_from_ckpt = None
    is_graph_model = True

    # Create trainer
    trainer = Trainer(model=model, data_module=data_module, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, hparams=hparams,
                    save_every_n_epoch=save_every_n_epoch, allow_log=allow_log, num_classes=num_classes, is_graph_model=is_graph_model,
                    resume_from_ckpt=resume_from_ckpt, max_epochs=hparams["max_epochs"], save_top_k=save_top_k)
    
    # Train model
    trainer.fit()

    # Test best model found
    trainer.test()

# Run script
if __name__ == "__main__":
    main()
