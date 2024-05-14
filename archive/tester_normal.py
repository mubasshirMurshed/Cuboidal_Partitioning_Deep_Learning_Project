from torch import nn
from datasets import *
from models import *
from trainer import *
from trainer.utils.logger import *
from dataModules.general_datamodule import *
import numpy as np
import matplotlib.pyplot as plt
import math
from torchvision.datasets import MNIST
from dataModules.ensembleDataModule import EnsembleDataModule, EnsembleDataModule2
from models.ensembleModel1 import EnsembleModel
from models.ensembleModel2 import EnsembleModel2

def main():
    num_classes = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()
    test_metrics = MetricCollection( {
        "Top 1 Accuracy" : MulticlassAccuracy(num_classes, top_k=1),
        "Top 2 Accuracy" : MulticlassAccuracy(num_classes, top_k=2),
        "Top 3 Accuracy" : MulticlassAccuracy(num_classes, top_k=3),
        "Confusion Matrix" : MulticlassConfusionMatrix(num_classes)
    }, compute_groups=False).to(device)


    data_module = EnsembleDataModule2(batch_size=100)
    model = EnsembleModel2(num_features=data_module.train_set.ds1.num_features)

    model_ckpt = r"saved\EnsembleDataModule2\XYCNA\EnsembleModel2\Run_ID__2024-02-19__21-15-29\checkpoints\epoch=40-val_loss=0.0617-val_acc=0.9874.pt"

    model.load_state_dict(torch.load(model_ckpt))
    model = model.to(device=device)
    model.eval() # puts the model in evaluation mode
    running_loss = 0
    predictions = np.zeros((len(data_module.test_set), 2), dtype=np.int64)
    offset = 0
    with torch.no_grad(): # save memory by not saving gradients which we don't need 
        for (i1, i2, i3, i4), labels in tqdm(data_module.test_dataloader(), leave=False):
            # Get batch of images and labels
            i1, i2, i3, i4, labels = i1.to(device), i2.to(device), i3.to(device), i4.to(device), labels.to(device)
            
            # Forward
            outputs = model(i1, i2, i3, i4)

            # Calculate metrics
            test_loss = loss_fn(outputs, labels) # calculates test_loss from model predictions and true labels
            running_loss += test_loss.item()

            # Update trackers
            test_metrics.update(outputs, labels)

            # Add to predictions in the form (index, label, predicted)
            batch_preds = torch.stack((labels.cpu(), outputs.argmax(dim=1).cpu()), dim=1).detach().numpy()
            predictions[offset : offset+len(labels), :] = batch_preds

            # Update offset
            offset += len(labels)

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