"""
PyTorch Dataset for recGPT.

@author Younggue
"""
import os
import requests

import numpy as np

import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    """
    Dataset for Generative pre-training.
    """
    def __init__(self, seq_list, block_size, val_holdout_list=None, vocab_size=None, train=True):
        super().__init__()
        self.seq_list = seq_list
        self.block_size = block_size
        self.val_holdout_list = val_holdout_list
        self.vocab_size = vocab_size
        self.train = train
        self.PAD_TOKEN_IDX = 0

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):
        seq = self.seq_list[index]
        if self.train:
            if len(seq) < 2:
                return None
            
            padding_len = (self.block_size + 1) - len(seq)
            # seq = [self.PAD_TOKEN_IDX] * padding_len + seq
            seq = seq + [self.PAD_TOKEN_IDX] * padding_len

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
            seq = [self.PAD_TOKEN_IDX] * padding_len + seq
            # Crop items to the last block_size tokens.
            x = torch.tensor(seq[-self.block_size:])
            return x


class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine Tuning.
    """

    def __init__(self, prompt_list, answer_list, block_size, val_holdout_list=None, vocab_size=None, train=True):
        super().__init__()
        self.prompt_list = prompt_list
        self.answer_list = answer_list
        self.block_size = block_size
        self.val_holdout_list = val_holdout_list
        self.vocab_size = vocab_size
        self.train = train
        self.PAD_TOKEN_IDX = 0

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, index):
        input_seq = self.prompt_list[index]
        answer_seq = self.answer_list[index] if self.answer_list is not None else None
           
        if len(input_seq) < 1:
            return None
        elif self.train and len(input_seq) < 2:
            return None
        
        input_padding_len = self.block_size - len(input_seq)
        #input_seq = [self.PAD_TOKEN_IDX] * input_padding_len + input_seq
        input_seq = input_seq + [self.PAD_TOKEN_IDX] * input_padding_len

        #x = torch.tensor(input_seq[-self.block_size:])
        x = torch.tensor(input_seq[:self.block_size])
        y = None
        labels = None
        if answer_seq:
            answer_padding_len = self.block_size - len(answer_seq)
            answer_seq = answer_seq + [self.PAD_TOKEN_IDX] * answer_padding_len
            y = torch.tensor(answer_seq[:self.block_size])

        if self.val_holdout_list:
            holdout_seq = self.val_holdout_list[index]
            labels = np.zeros(self.vocab_size)
            for idx, _ in enumerate(labels):
                if idx in holdout_seq:
                    labels[idx] = 1

        if y is not None and labels is not None:
            return x, y, torch.tensor(labels)
        elif y is not None:
            return x, y
        else:
            return x
