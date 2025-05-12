import torch
from torch.utils.data import Dataset

class GlucoseDataset(Dataset):
    def __init__(self, samples, seq_len, mu, std):
        self.samples = samples
        self.seq_len = seq_len
        self.mu = mu
        self.std = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = s[:self.seq_len, :]
        y = (s[self.seq_len, 0] - self.mu) / self.std # predict *next* mg/dl
        return torch.tensor(x), torch.tensor(y).unsqueeze(-1)
