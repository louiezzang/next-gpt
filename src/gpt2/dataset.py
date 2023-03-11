"""
PyTorch Dataset for GPT.

@author Younggue
"""
import os
import requests

import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):

    def __init__(self, seq_list, block_size, train=True):
        super().__init__()
        self.seq_list = seq_list
        self.block_size = block_size
        self.train = train

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):
        seq = self.seq_list[index]
        if self.train:
            if len(seq) < self.block_size + 1:
                return None
            x = torch.tensor(seq[:self.block_size])
            y = torch.tensor(seq[1:self.block_size+1])
            return x, y
        else:
            # Crop items to the last block_size tokens.
            x = torch.tensor(seq[:-self.block_size])
            return x
