# %% Imports
from datasets import MNISTGraphDataset_Old, MNISTGraphDataset
from time import perf_counter

# %% Instantiate normal dataset
start = perf_counter()
ds1 = MNISTGraphDataset_Old(csv_file_dir=r"data\mnistNew\mnistTrain128.csv", length=100)
end = perf_counter()
execution_time = (end - start)
print(f"Normal dataset with manual adjacency conversion: {execution_time}")

# %% Instantiate InMemoryDataset
start = perf_counter()
ds2 = MNISTGraphDataset(root="data/mnistNew", filename="mnistTrain128", length=100)
end = perf_counter()
execution_time = (end - start)
print(f"InMemoryDataset with tensor funcs for conversion: {execution_time}")
# %% Get image