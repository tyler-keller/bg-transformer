import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset

class GlucoseDataset(Dataset):
    def __init__(self, sequences, targets, mu=None, std=None):
        self.sequences = sequences
        self.targets = targets
        
        # if mu/std not provided, calculate them (usually for training set)
        if mu is None:
            self.mu = np.mean(targets)
            self.std = np.std(targets)
        else:
            self.mu = mu
            self.std = std

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = self.sequences[idx]
        y_raw = self.targets[idx]
        
        # standard scaling for target
        y_norm = (y_raw - self.mu) / self.std
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y_norm, dtype=torch.float32).unsqueeze(-1)

def load_and_window_data(data_dir, seq_len):
    """reads all csvs in dir and creates sliding windows."""
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")])
    x_list, y_list = [], []
    
    for f in all_files:
        # load raw array
        data = pd.read_csv(f).to_numpy(dtype=np.float32)
        
        if len(data) > seq_len:
            # vectorize windowing could be faster, but loop is safe for variable file lengths
            for i in range(len(data) - seq_len):
                # features: all columns. target: column 0 (mg/dl) at next step
                window = data[i : i + seq_len]
                target = data[i + seq_len, 0] 
                x_list.append(window)
                y_list.append(target)
                
    return np.array(x_list), np.array(y_list)