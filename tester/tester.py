from torch import nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data.datasets import *
from models.mnistGAT2 import MNIST_GAT2
import numpy as np
import matplotlib.pyplot as plt
import math
from torchvision.datasets import MNIST
from data.datamodules import General_DataModule
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchmetrics import MetricCollection

def main():
    num_classes = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()
    test_metrics = MetricCollection( {
        "Top 1 Accuracy" : MulticlassAccuracy(num_classes),
        "Top 2 Accuracy" : MulticlassAccuracy(num_classes, top_k=2),
        "Top 3 Accuracy" : MulticlassAccuracy(num_classes, top_k=3),
        "Confusion Matrix" : MulticlassConfusionMatrix(num_classes)
    }, compute_groups=False).to(device)

    data_module = General_DataModule("mnist", num_segments=64, batch_size=100, mode="CP", x_centre=True, y_centre=True, colour=True, num_pixels=True, angle=True)
    model = MNIST_GAT2(num_features=data_module.train_set.num_features)

    model_ckpt = r"saved\Group_DataModule2\XYCNA\MNIST_GAT2\Run_ID__2024-02-20__23-49-34\checkpoints\epoch=18-val_loss=0.0713-val_acc=0.9771.pt"
    model.load_state_dict(torch.load(model_ckpt))
    model = model.to(device=device)
    model.eval() # puts the model in evaluation mode
    running_loss = 0
    predictions = np.zeros((len(data_module.test_set), 2), dtype=np.int64)
    offset = 0
    with torch.no_grad(): # save memory by not saving gradients which we don't need 
        for batch in tqdm(data_module.test_dataloader(), leave=False):
            # Get batch of images and labels
            batch = batch.to(device) # put the data on the GPU
            
            # Forward
            outputs = model(batch.x, batch.edge_index, batch.batch) # passes image to the model, and gets an ouput which is the class probability prediction

            # Calculate metrics
            test_loss = loss_fn(outputs, batch.y) # calculates test_loss from model predictions and true labels
            running_loss += test_loss.item()

            # Update trackers
            test_metrics.update(outputs, batch.y)

            # Add to predictions in the form (index, label, predicted)
            batch_preds = torch.stack((batch.y.cpu(), outputs.argmax(dim=1).cpu()), dim=1).detach().numpy()
            predictions[offset : offset+batch.num_graphs, :] = batch_preds

            # Update offset
            offset += batch.num_graphs

        avg_test_loss = running_loss/len(data_module.test_dataloader()) # return average test loss

    test_metric_results = test_metrics.compute()
    test_metrics.reset()
    print('-' * 80)
    print("Test Dataset Results:")
    print('-' * 80)
    print(f"Loss: {avg_test_loss:.5}")
    print(f"Top 1 Accuracy: {test_metric_results['Top 1 Accuracy']:.2%}")
    print(f"Top 2 Accuracy: {test_metric_results['Top 2 Accuracy']:.2%}")
    print(f"Top 3 Accuracy: {test_metric_results['Top 3 Accuracy']:.2%}")
    print('-' * 80)
    print(test_metric_results["Confusion Matrix"].to("cpu").numpy())
    print('-' * 80)

    # Do mislabelled processing
    mislabelled = []
    for i in range(len(predictions)):
        if predictions[i, 0] != predictions[i, 1]:
            mislabelled.append((i, predictions[i, 0].item(), predictions[i, 1].item()))

    print("Mislabelled MNIST Test images: " + str(list(zip(*mislabelled))[0]))

    # Only view first 50
    mislabelled = mislabelled[:50]

    # Plot all MNIST images in a grid with caption of label vs predicted
    mnist = MNIST(root="data/mnistPytorch/", train=False)
    NUM_COLUMNS = 10
    NUM_ROWS = math.ceil(len(mislabelled) / NUM_COLUMNS)
    fig, axes = plt.subplots(NUM_ROWS, NUM_COLUMNS)
    for i in range(len(mislabelled)):
        data = mislabelled[i]
        ax = axes[i // NUM_COLUMNS][i % NUM_COLUMNS]
        ax.imshow(mnist[data[0]][0], cmap="gray")
        ax.set_title(f"Truth: {data[1]}, Pred: {data[2]}", fontsize=5)
        ax.axis("off")
    
    plt.show()


if __name__ == "__main__":
    main()