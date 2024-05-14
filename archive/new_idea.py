import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import torch
from torch import nn
from trainer import Trainer
from dataModules.ensembleDataModule import EnsembleDataModule2
from models.ensembleModel2 import EnsembleModel2

hparams = {
    "max_epochs" : 100,
    "learning_rate" : 0.0001,
    "batch_size" : 256
}

def main():
    num_classes = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dm = EnsembleDataModule2(hparams["batch_size"])
    model = EnsembleModel2(num_features=dm.train_set.ds1.num_features)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hparams["learning_rate"])

    allow_log = True
    save_every_n_epoch = 1
    resume_from_ckpt= None
    is_graph_model = False

    trainer = Trainer(model=model, data_module=dm, loss_fn=loss_fn, optimizer=optimizer, hparams=hparams,
                    save_every_n_epoch=save_every_n_epoch, allow_log=allow_log, num_classes=num_classes, is_graph_model=is_graph_model,
                    resume_from_ckpt=resume_from_ckpt)
    
    trainer.fit()
    trainer.test()

if __name__ == "__main__":
    main()