import numpy as np
import pdb  # Python Debugger, used for interactive debugging
from torch.utils.data import Dataset, DataLoader  # PyTorch utilities for handling datasets
import torch  # PyTorch library for tensor computations
import matplotlib.pyplot as plt  # Library for plotting graphs
import os  # Library for interacting with the operating system
from tqdm import tqdm  # Library for displaying progress bars

# Definition of a custom dataset class for Backward-Facing Step (BFS) data
class bfs_dataset(Dataset):
    def __init__(self,
                 data_location=['./data0.npy', './data1.npy'],  # Default locations for data files
                 trajec_max_len=50,  # Maximum length of the data sequence
                 start_n=0,  # Starting index for data selection
                 n_span=510):  # Total number of data points to include
        assert n_span > trajec_max_len  # Ensuring the span of data is larger than the maximum sequence length
        self.start_n = start_n
        self.n_span = n_span
        self.trajec_max_len = trajec_max_len

        # Loading and concatenating data from specified files
        solution0 = np.load(data_location[0], allow_pickle=True)
        # solution = np.load(data_location[0], allow_pickle=True)
        solution1 = np.load(data_location[1], allow_pickle=True)
        solution = np.concatenate([solution0, solution1], axis=0)
        self.solution = torch.from_numpy(solution[start_n:start_n + n_span])  # Converting to a PyTorch tensor
        del solution, solution1
    # Returns the length of the dataset
    def __len__(self):
        return self.n_span - self.trajec_max_len

    # Returns a specific item from the dataset given an index
    def __getitem__(self, index):
        return self.solution[index:index + self.trajec_max_len]


# Main execution block
if __name__ == '__main__':
    dset = bfs_dataset()  # Creating an instance of the BFS dataset
    dloader = DataLoader(dataset=dset, batch_size=20, shuffle=True)  # DataLoader for batching and shuffling BFS data

    # Iterating over the DataLoader
    for iteration, batch in enumerate(dloader):
        print(iteration)  # Printing the iteration number
        print('Do something!')  # Placeholder for processing the BFS data
        pdb.set_trace()  # Pausing the program for debugging purposes