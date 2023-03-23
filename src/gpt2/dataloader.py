"""
Pytorch DataLoader for GPT dataset.

@author Younggue Bae
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
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

    def get_pytorch_dataloaders(self, rank, world_size, shuffle=True):
        train_loader = self.get_train_loader(rank, world_size, shuffle)
        val_loader = self.get_val_loader(rank, world_size, shuffle)
        test_loader = self.get_test_loader(rank, world_size, shuffle)
        return train_loader, val_loader, test_loader

    def get_train_loader(self, rank, world_size, shuffle=True):
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
                                    shuffle=shuffle,
                                    collate_fn=self.collate_fn,
                                    drop_last=self.drop_last)
        return dataloader

    def get_val_loader(self, rank, world_size, shuffle=True):
        return self._get_eval_loader(mode="val", rank=rank, world_size=world_size, shuffle=shuffle)

    def get_test_loader(self, rank, world_size, shuffle=True):
        return self._get_eval_loader(mode="test", rank=rank, world_size=world_size, shuffle=shuffle)

    def _get_eval_loader(self, mode, rank, world_size, shuffle):
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
                                    shuffle=shuffle,
                                    collate_fn=self.collate_fn,
                                    drop_last=self.drop_last)
        return dataloader


class BatchSampler(Sampler):
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        for batch in self.batches:
             yield batch

    def __len__(self):
        return len(self.batches)


class GPTRandomSampleDataloader(object):
    def __init__(self, dataset, block_size, batch_size, meta=None):
        """
        Random sample data loader from the entire dataset.

        Args:
            dataset (dict): The train dataset which contains the entire ids from source data or torch.utils.data.Dataset (key: 'train', 'val')
            block_size (int): The block size (eg. max token length)
            batch_size (int): The batch size
            meta (dict, optional): The extra metadata (eg. vocab_size, itos, stoi). Defaults to None.
        """
        self.train = dataset["train"] if "train" in dataset else None
        self.val = dataset["val"] if "val" in dataset else None

        
        if (self.train is not None and isinstance(self.train, (list, np.ndarray, np.generic))) or \
            (self.val is not None and isinstance(self.val, (list, np.ndarray, np.generic))):
            self.dataset_type = "list"
            if self.train is not None and isinstance(self.train, (list)):
                self.train = np.array(self.train, dtype=np.uint16)
            if self.val is not None and isinstance(self.val, (list)):
                self.val = np.array(self.val, dtype=np.uint16)
        elif (self.train is not None and isinstance(self.train, Dataset)) or \
            (self.val is not None and isinstance(self.val, Dataset)):
            self.dataset_type = "dataset"
        else:
            raise ValueError(f"Not supported data type: {type(self.train)}")

        self.block_size = block_size
        self.batch_size = batch_size
        self.meta = meta

    def get_batch(self, split):
        if self.dataset_type == "list":
            data = self.train if split == "train" else self.val
            ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        else:
            data = self.train if split == "train" else self.val
            ix = torch.randint(len(data), (self.batch_size,))
            # batch_sampler = BatchSampler([[1, 2, 3], [5, 6, 7], [4, 2, 1]])
            batch_sampler = BatchSampler([ix.tolist()])
            dataloader = DataLoader(dataset=data, batch_sampler=batch_sampler)
            for batch in dataloader: # dataloader size is one!
                x, y = batch
        
        return x, y
