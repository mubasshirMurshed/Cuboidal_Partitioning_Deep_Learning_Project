from torch_geometric.data import Dataset
from datasets.graph_dataset import MNISTGraphDataset_CSV

class GroupDataset(Dataset):
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
                                        mode="CP",
                                        num_cuboids=32,
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
        combined_graph = first_res.concat(second_res)
        
        # Fix edges
        first_res_num_edges = first_res.edge_index.shape[1]
        combined_graph.edge_index[:, first_res_num_edges:] += first_res.num_nodes

        # Fix label
        combined_graph.y = first_res.y

        return combined_graph
    
class GroupDataset2(Dataset):
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
        combined_graph = first_res.concat(second_res)
        
        # Fix edges
        first_res_num_edges = first_res.edge_index.shape[1]
        combined_graph.edge_index[:, first_res_num_edges:] += first_res.num_nodes

        # Fix label
        combined_graph.y = first_res.y

        return combined_graph