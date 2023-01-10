import torch
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
 
# Load the MNISTSuperpixel dataset
data = MNISTSuperpixels(root="data/mnistSuperpixel")

embedding_size = 64
class GCN(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(data.num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer
        self.out = Linear(embedding_size*2, data.num_classes)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)

        # Apply a final (linear) classifier.
        out = self.out(hidden)
        return out

model = GCN()
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
# Use GPU for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Wrap data in a data loader
data_size = len(data)
NUM_GRAPHS_PER_BATCH = 64

train_data , test_data = random_split(data, [0.6, 0.4])

train_loader = DataLoader(train_data, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(test_data, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)

def fit(epochs, model, train_loader, test_loader, optimizer, device):
    for epoch in range(epochs):
        # Run training cycle
        train_loss = train(model, train_loader, optimizer, device)
        
        # Run validation cycle
        val_loss, accuracy = validation(model, test_loader, device)

        # Print metrics to console
        print("Epoch: {}/{}, Training Loss: {:.3f}, Val Loss: {:.3f}, Val Accuracy: {:.2f}%".format(epoch+1, epochs, train_loss, val_loss, accuracy*100))
        print('-' * 20)
            
def train(model, train_loader, optimizer, device):
        """
        Performs one training loop over the training data loader.
        """
        # Put the model in training mode
        model.train()
        running_loss = 0
        for batch in tqdm(train_loader, leave=False):
            # Get batch of images and labels
            batch = batch.to(device) # puts the data on the GPU

            # Forward                                         
            optimizer.zero_grad() # clear the gradients in model parameters
            outputs = model(batch.x.float(), batch.edge_index, batch.batch) # forward pass and get predictions

            # Backward
            loss = F.cross_entropy(outputs, batch.y) # calculate loss
            loss.backward() # calculates gradient w.r.t to loss for all parameters in model that have requires_grad=True
            
            # Update weights
            optimizer.step() # iterate over all parameters in the model with requires_grad=True and update their weights.

            running_loss += loss.item() # sum total loss in current epoch for print later

        return running_loss/len(train_loader) # returns the average training loss for the epoch

def validation(model, test_loader, device):
    """
    Performs one validation loop over the validation data loader.
    """
    model.eval() # puts the model in validation mode
    running_loss = 0
    total = 0
    correct = 0
    
    with torch.no_grad(): # save memory by not saving gradients which we don't need 
        for batch in tqdm(test_loader, leave=False):
            # Get batch of images and labels
            batch = batch.to(device) # put the data on the GPU
            
            # Forward
            outputs = model(batch.x.float(), batch.edge_index, batch.batch) # passes image to the model, and gets an ouput which is the class probability prediction

            # Calculate metrics
            val_loss = F.cross_entropy(outputs, batch.y) # calculates val_loss from model predictions and true labels
            running_loss += val_loss.item()
            _, predicted = torch.max(outputs, 1) # turns class probability predictions to class labels
            total += batch.y.size(0) # sums the number of predictions
            correct += (predicted == batch.y).sum().item() # sums the number of correct predictions

        return running_loss/len(test_loader), correct/total # return average validation loss, accuracy

fit(10, model, train_loader, test_loader, optimizer, device)