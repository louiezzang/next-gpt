"""
Pytorch DataLoader for GPT dataset.

@author Younggue Bae
"""
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class GPTDataloader(object):
    def __init__(self, args, dataset, collate_fn=None):
        self.args = args
        self.train = dataset["train"]  # Mandatory for training
        self.val = dataset["val"] if "val" in dataset else None  # Optional
        self.test = dataset["test"] if "test" in dataset else None  # Optional
        self.collate_fn = collate_fn if collate_fn else self.default_collate_fn

        self.train_batch_size = args.train_batch_size  # Mandatory for training
        self.val_batch_size = args.val_batch_size if hasattr(args, "val_batch_size") else self.train_batch_size
        self.test_batch_size = args.test_batch_size if hasattr(args, "test_batch_size") else self.train_batch_size

        seed = args.dataloader_random_seed if hasattr(args, "dataloader_random_seed") else 1
        self.rng = random.Random(seed)
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

