from torch_geometric.data import Dataset
from datasets.graph_dataset import MNISTGraphDataset_CSV

class EnsembleDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.split = split
        self.ds1 = MNISTGraphDataset_CSV(root=self.root,
                                        split=self.split,
                                        mode="CP",
                                        num_cuboids=64,
                                        x_centre=True,
                                        y_centre=True,
                                        colour=True,
                                        num_pixels=True,
                                        angle=True
                                        )
        
        self.ds2 = MNISTGraphDataset_CSV(root=self.root,
                                        split=self.split,
                                        mode="CP45",
                                        num_cuboids=64,
                                        x_centre=True,
                                        y_centre=True,
                                        colour=True,
                                        num_pixels=True,
                                        angle=True
                                        )
        self.ablation_code = self.ds1.ablation_code
        super().__init__(root)
        
    def len(self):
        return len(self.ds1)

    def get(self, idx):
        first_res = self.ds1[idx]
        second_res = self.ds2[idx]
        label = self.ds1[idx].y
        return (first_res, second_res), label.item()
    

class EnsembleDataset2(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.split = split
        self.ds1 = MNISTGraphDataset_CSV(root=self.root,
                                        split=self.split,
                                        mode="CP",
                                        num_cuboids=64,
                                        x_centre=True,
                                        y_centre=True,
                                        colour=True,
                                        num_pixels=True,
                                        angle=True
                                        )
        
        self.ds2 = MNISTGraphDataset_CSV(root=self.root,
                                        split=self.split,
                                        mode="CP45",
                                        num_cuboids=64,
                                        x_centre=True,
                                        y_centre=True,
                                        colour=True,
                                        num_pixels=True,
                                        angle=True
                                        )
        
        self.ds3 = MNISTGraphDataset_CSV(root=self.root,
                                        split=self.split,
                                        mode="CP22_5",
                                        num_cuboids=64,
                                        x_centre=True,
                                        y_centre=True,
                                        colour=True,
                                        num_pixels=True,
                                        angle=True
                                        )
        
        self.ds4 = MNISTGraphDataset_CSV(root=self.root,
                                        split=self.split,
                                        mode="CP22_5neg",
                                        num_cuboids=64,
                                        x_centre=True,
                                        y_centre=True,
                                        colour=True,
                                        num_pixels=True,
                                        angle=True
                                        )

        self.ablation_code = self.ds1.ablation_code
        super().__init__(root)
        
    def len(self):
        return len(self.ds1)

    def get(self, idx):
        first_res = self.ds1[idx]
        second_res = self.ds2[idx]
        third_res = self.ds3[idx]
        fourth_res = self.ds4[idx]
        label = self.ds1[idx].y
        return (first_res, second_res, third_res, fourth_res), label.item()