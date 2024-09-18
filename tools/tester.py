import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from torch import nn
from models.GAT_Modelv2 import GAT_Modelv2
import matplotlib.pyplot as plt
import math
from data.datamodules import Graph_DataModule_CSV
from trainer import Trainer
from data.data_classes import MyMNIST, MyCIFAR_10, MyMedMNIST, MyOmniglot
from enums import Partition
import numpy as np

class Tester:
    def __init__(self, data_module: Graph_DataModule_CSV, model, loss_fn):
        # Create trainer
        self.data_module = data_module
        self.trainer = Trainer(model=model, data_module=data_module, loss_fn=loss_fn, is_graph_model=True)

    def show_mislabelled(self, model_ckpt):
        # Test best model
        mislabelled = self.trainer.test(model_ckpt)

        # Limit how many to view
        TOTAL_VIEW = 50
        mislabelled = mislabelled[:TOTAL_VIEW, :]

        # Plot all mislabelled images in a grid
        NUM_COLUMNS = 10
        NUM_ROWS = math.ceil(TOTAL_VIEW / NUM_COLUMNS)
        _, axes = plt.subplots(NUM_ROWS, NUM_COLUMNS)
        for i in range(len(mislabelled)):
            data = mislabelled[i]
            ax = axes[i // NUM_COLUMNS][i % NUM_COLUMNS]
            img = np.asarray(self.data_module.test_set[data[2]][0])
            if len(img.shape) == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.set_title(f"Truth: {data[0]}, Pred: {data[1]}", fontsize=5)
            ax.axis("off")
        plt.show()

    def compare_mislabelled(self, *model_ckpts):
        # Get the different log directories where the mislabel information is
        log_dirs = [str(Path(model_ckpt).parent.parent) for model_ckpt in model_ckpts]

        # Load in mislabel files
        mislabelled_sets = [set(np.load(log_dir + "/mislabelled.npy")[:, 2].flatten()) for log_dir in log_dirs]

        # Print comparison information
        for i in range(len(model_ckpts)):
            print(f"Number of incorrect predictions for {model_ckpts[i]}\n{len(mislabelled_sets[i])}\n")

        first = mislabelled_sets[0]
        rest = mislabelled_sets[1:]
        shared = first.intersection(*rest)
        print(f"There are {len(shared)} items that are mislabelled across all provided model checkpoints.")
        print(shared)

if __name__ == "__main__":
    # Create data module
    features = {"x_center":True, "y_center":True, "colour":True, "width":True, "height":True}
    data_module = Graph_DataModule_CSV(
        dataset=MyMNIST(),
        num_segments=64,
        batch_size=100,
        mode=Partition.CuPID,
        num_workers=1,
        features=features
    )

    # Instantiate model
    model = GAT_Modelv2(num_features=data_module.num_features, num_classes=data_module.num_classes)

    # Initialise loss function
    loss_fn = nn.CrossEntropyLoss()

    # Model checkpoint to test
    model_ckpt = r"saved\MNIST_CP_64_DataModule\XYCWH\GAT_Modelv2\Run_ID__2024-09-19__00-27-46\checkpoints\best.pt"

    tester = Tester(data_module, model, loss_fn)
    tester.show_mislabelled(model_ckpt)
    tester.compare_mislabelled(model_ckpt, 
                               r"saved\MNIST_CP_64_DataModule\XYCWH\GAT_Modelv2\Run_ID__2024-09-19__01-03-48\checkpoints\best.pt",
                               r"saved\MNIST_CP_64_DataModule\XYCWH\GAT_Modelv2\Run_ID__2024-09-19__01-09-24\checkpoints\best.pt")