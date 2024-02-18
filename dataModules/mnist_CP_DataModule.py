from dataModules.dataModule import DataModule
from torch_geometric.loader import DataLoader
from datasets import MNISTGraphDataset_CSV


class MNIST_CP_DataModule(DataModule):
    """
    A data module for the cuboidal graph dataset.
    """
    def __init__(self, num_cuboids: int, batch_size: int, x_centre=False,
                y_centre=False, colour=False, num_pixels=False, angle=False,
                width=False, height=False):
        """
        Save attributes.

        Args:
        - batch_size: int
            - How many data samples per batch to be loaded
        """
        self.num_cuboids = num_cuboids
        self.x_centre = x_centre
        self.y_centre = y_centre
        self.colour = colour
        self.num_pixels = num_pixels
        self.angle = angle
        self.width = width
        self.height = height
        super().__init__(batch_size, DataLoader)
        

    def setup(self):
        """
        Instantiate datasets for training and validation.
        """
        self.train_set = MNISTGraphDataset_CSV(root="data/",
                                              split="Train",
                                              mode="CP",
                                              num_cuboids=self.num_cuboids,
                                              x_centre=self.x_centre,
                                              y_centre=self.y_centre,
                                              colour=self.colour,
                                              num_pixels=self.num_pixels,
                                              angle=self.angle,
                                              width=self.width,
                                              height=self.height
                                             )

        self.val_set = MNISTGraphDataset_CSV(root="data/",
                                            split="Validation",
                                            mode="CP",
                                            num_cuboids=self.num_cuboids,
                                            x_centre=self.x_centre,
                                            y_centre=self.y_centre,
                                            colour=self.colour,
                                            num_pixels=self.num_pixels,
                                            angle=self.angle,
                                            width=self.width,
                                            height=self.height
                                           )
        
        self.test_set = MNISTGraphDataset_CSV(root="data/",
                                            split="Test",
                                            mode="CP",
                                            num_cuboids=self.num_cuboids,
                                            x_centre=self.x_centre,
                                            y_centre=self.y_centre,
                                            colour=self.colour,
                                            num_pixels=self.num_pixels,
                                            angle=self.angle,
                                            width=self.width,
                                            height=self.height
                                           )