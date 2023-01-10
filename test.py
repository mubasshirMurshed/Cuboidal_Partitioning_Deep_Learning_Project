import pandas as pd
import torch

# Read in data file and save attributes
data = pd.read_csv("data/mnistNew/mnistTest128.csv", nrows=5)

# Fill in self.dataset with the appropriate reshapes of the data
dataset = [0]*len(data)

x = torch.zeros(data.values[2][0], 5)
print(x.shape)