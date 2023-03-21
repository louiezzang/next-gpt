"""
PyTorch Dataset for GPT.

@author Younggue
"""
import os
import requests

import numpy as np

import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):

    def __init__(self, seq_list, block_size, val_holdout_list=None, vocab_size=None, train=True):
        super().__init__()
        self.seq_list = seq_list
        self.block_size = block_size
        self.val_holdout_list = val_holdout_list
        self.vocab_size = vocab_size
        self.train = train
        self.PADDING_TOKEN_IDX = 0

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):
        seq = self.seq_list[index]
        if self.train:
            if len(seq) < 2:
                return None
            
            padding_len = (self.block_size + 1) - len(seq)
            seq = [self.PADDING_TOKEN_IDX] * padding_len + seq

            # x = torch.tensor(seq[:self.block_size])
            # y = torch.tensor(seq[1:self.block_size+1])
            # Crop items to the last block_size tokens.
            x = torch.tensor(seq[-self.block_size-1:-1])
            y = torch.tensor(seq[-self.block_size:])

            if self.val_holdout_list:
                holdout_seq = self.val_holdout_list[index]
                labels = np.zeros(self.vocab_size)
                for idx, _ in enumerate(labels):
                    if idx in holdout_seq:
                        labels[idx] = 1

                return x, y, torch.tensor(labels)
            return x, y
        else:
            if len(seq) < 1:
                return None
            
            padding_len = self.block_size - len(seq)
            seq = [self.PADDING_TOKEN_IDX] * padding_len + seq
            # Crop items to the last block_size tokens.
            x = torch.tensor(seq[-self.block_size:])
            return x
