from .dataModule import DataModule
from torch_geometric.loader.dataloader import DataLoader
from datasets.ensembleDataset import EnsembleDataset, EnsembleDataset2

class EnsembleDataModule(DataModule):
    def __init__(self, batch_size: int):
        super().__init__(batch_size, DataLoader)

    def setup(self):
        self.train_set = EnsembleDataset(root="data/", split="Train")

        self.val_set = EnsembleDataset(root="data/", split="Validation")
        
        self.test_set = EnsembleDataset(root="data/", split="Test")

class EnsembleDataModule2(DataModule):
    def __init__(self, batch_size: int):
        super().__init__(batch_size, DataLoader)

    def setup(self):
        self.train_set = EnsembleDataset2(root="data/", split="Train")

        self.val_set = EnsembleDataset2(root="data/", split="Validation")
        
        self.test_set = EnsembleDataset2(root="data/", split="Test")

