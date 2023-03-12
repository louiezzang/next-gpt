"""
Pytorch DataLoader for GPT dataset.

@author Younggue Bae
"""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class GPTDataloader(object):
    def __init__(self, args, dataset, collate_fn=None):
        self.args = args
        self.train = dataset["train"] if "train" in dataset else None  # Optional
        self.val = dataset["val"] if "val" in dataset else None  # Optional
        self.test = dataset["test"] if "test" in dataset else None  # Optional
        self.collate_fn = collate_fn if collate_fn else self.default_collate_fn

        self.train_batch_size = args.train_batch_size if hasattr(args, "train_batch_size") else 12
        self.val_batch_size = args.val_batch_size if hasattr(args, "val_batch_size") else self.train_batch_size
        self.test_batch_size = args.test_batch_size if hasattr(args, "test_batch_size") else self.train_batch_size

        self.drop_last = args.drop_last if hasattr(args, "drop_last") else False

    @staticmethod
    def default_collate_fn(batch):
        # Skip the error data.
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    def get_pytorch_dataloaders(self, rank, world_size):
        train_loader = self.get_train_loader(rank, world_size)
        val_loader = self.get_val_loader(rank, world_size)
        test_loader = self.get_test_loader(rank, world_size)
        return train_loader, val_loader, test_loader

    def get_train_loader(self, rank, world_size):
        dataset = self.train
        if world_size > 0:
            sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size)
            dataloader = DataLoader(dataset,
                                    batch_size=self.train_batch_size,
                                    shuffle=False,
                                    collate_fn=self.collate_fn,
                                    sampler=sampler,
                                    drop_last=self.drop_last)
        else:
            dataloader = DataLoader(dataset,
                                    batch_size=self.train_batch_size,
                                    shuffle=True,
                                    collate_fn=self.collate_fn,
                                    drop_last=self.drop_last)
        return dataloader

    def get_val_loader(self, rank, world_size):
        return self._get_eval_loader(mode="val", rank=rank, world_size=world_size)

    def get_test_loader(self, rank, world_size):
        return self._get_eval_loader(mode="test", rank=rank, world_size=world_size)

    def _get_eval_loader(self, mode, rank, world_size):
        batch_size = self.val_batch_size if mode == "val" else self.test_batch_size
        dataset = self.val if mode == "val" else self.test
        if dataset is None:
            return None

        if world_size > 0:
            sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size)
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    collate_fn=self.collate_fn,
                                    sampler=sampler,
                                    drop_last=self.drop_last)
        else:
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    collate_fn=self.collate_fn,
                                    drop_last=self.drop_last)
        return dataloader



class GPTRandomAccessDataloader(object):
    def __init__(self, dataset, block_size, batch_size, meta=None):
        """
        Random access data loader from the entire dataset.

        Args:
            dataset (dict): The dataset which contains the entire ids from source data(key: train_ids, val_ids)
            block_size (int): The block size (eg. max token length)
            batch_size (int): The batch size
            meta (dict, optional): The extra metadata (eg. vocab_size, itos, stoi). Defaults to None.
        """
        self.train_ids = dataset["train_ids"] if "train_ids" in dataset else None
        self.val_ids = dataset["val_ids"] if "val_ids" in dataset else None

        self.block_size = block_size
        self.batch_size = batch_size
        self.meta = meta

    def get_batch(self, split):
        data = self.train_ids if split == "train" else self.val_ids
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        
        return x, y
