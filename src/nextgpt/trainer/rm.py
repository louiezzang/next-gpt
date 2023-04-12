from abc import ABC
from datetime import datetime
from typing import Optional

import pandas as pd
import torch
from torch import nn
import torch.distributed as dist
from torch.optim import Adam, AdamW, Optimizer, lr_scheduler
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .strategies import Strategy
from ..models import LogExpLoss, LogSigLoss
from .utils import is_rank_0


class RewardModelTrainer(ABC):
    """
        Trainer to use while training reward model.
    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        train_dataset (Dataset): the dataset to use for training
        valid_dataset (Dataset): the dataset to use for validation
        eval_dataset (Dataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        loss_fn (str, defaults to 'log_sig'): the loss function name to use for training ('log_sig', 'log_exp')
        lr (float, defaults to 5e-5): the learning rate
    """

    def __init__(
        self,
        model,
        strategy: Strategy,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        eval_dataset: Dataset,
        batch_size: int = 1,
        max_epochs: int = 1,
        loss_fn: str = "log_sig",
        lr: float = 5e-5
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.lr = lr
        train_sampler = None

        if dist.is_initialized() and dist.get_world_size() > 1:
            train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=42, drop_last=True)
        self.train_dataloader = DataLoader(train_dataset,
                                           shuffle=(train_sampler is None),
                                           sampler=train_sampler,
                                           batch_size=batch_size)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

        self.model = strategy.setup_model(model)
        if "DDP" in str(self.strategy):
            self.model = self.model.module

        self.loss_fn = self.get_loss_fn(loss_fn)
        optim = self.get_optimizer()
        self.optimizer = strategy.setup_optimizer(optim, self.model)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, self.train_dataloader.__len__() // 100)

    def get_optimizer(self) -> Optimizer:
        # optim = Adam(self.model.parameters(), lr=self.lr)

        # TODO: Need to define as arguments.
        # weight_decay (`float`, *optional*, defaults to 0): The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights.
        # adam_beta1 (`float`, *optional*, defaults to 0.9): The beta1 hyperparameter for the [`AdamW`] optimizer.
        # adam_beta2 (`float`, *optional*, defaults to 0.999): The beta2 hyperparameter for the [`AdamW`] optimizer.
        # adam_epsilon (`float`, *optional*, defaults to 1e-8):The epsilon hyperparameter for the [`AdamW`] optimizer.
        weight_decay = 0.0
        adam_beta1 = 0.9
        adam_beta2 = 0.999
        adam_epsilon = 1e-8
        optim = AdamW(self.model.parameters(), lr=self.lr, betas=(adam_beta1, adam_beta2), eps=adam_epsilon, weight_decay=weight_decay)

        return optim
    
    def get_loss_fn(self, loss_fn_name: str) -> nn.Module:
        if loss_fn_name == "log_sig":
            loss_fn = LogSigLoss()
        elif loss_fn_name == "log_exp":
            loss_fn = LogExpLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn_name}")

        return loss_fn
    
    def eval_acc(self, dataloader):
        dist = 0
        on = 0
        cnt = 0
        self.model.eval()
        with torch.no_grad():
            for chosen_ids, c_mask, reject_ids, r_mask in dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                reject_reward = self.model(reject_ids, attention_mask=r_mask)
                for i in range(len(chosen_reward)):
                    cnt += 1
                    if chosen_reward[i] > reject_reward[i]:
                        on += 1
                dist += (chosen_reward - reject_reward).mean().item()
            dist_mean = dist / len(dataloader)
            acc = on / cnt
        self.model.train()
        return dist_mean, acc

    def fit(self):
        time = datetime.now()
        epoch_bar = tqdm(range(self.epochs), desc='Train epoch', disable=not is_rank_0())
        for epoch in range(self.epochs):
            step_bar = tqdm(range(self.train_dataloader.__len__()),
                            desc='Train step of epoch %d' % epoch,
                            disable=not is_rank_0())
            # train
            self.model.train()
            cnt = 0
            acc = 0
            dist = 0
            for chosen_ids, c_mask, reject_ids, r_mask in self.train_dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                reject_reward = self.model(reject_ids, attention_mask=r_mask)
                loss = self.loss_fn(chosen_reward, reject_reward)
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer)
                self.optimizer.zero_grad()
                cnt += 1
                if cnt == 100:
                    self.scheduler.step()
                    dist, acc = self.eval_acc(self.valid_dataloader)
                    cnt = 0
                    if is_rank_0():
                        log = pd.DataFrame([[step_bar.n, loss.item(), dist, acc]],
                                           columns=['step', 'loss', 'dist', 'acc'])
                        log.to_csv('log_%s.csv' % time, mode='a', header=False, index=False)
                step_bar.update()
                # step_bar.set_postfix({'dist': dist, 'acc': acc})
                step_bar.set_postfix({'loss': loss.item(), 'dist': dist, 'acc': acc})

            # eval
            dist, acc = self.eval_acc(self.eval_dataloader)
            if is_rank_0():
                log = pd.DataFrame([[step_bar.n, loss.item(), dist, acc]], columns=['step', 'loss', 'dist', 'acc'])
                log.to_csv('log.csv', mode='a', header=False, index=False)
            epoch_bar.update()
            step_bar.set_postfix({'dist': dist, 'acc': acc})
            step_bar.close()

    def save_model(self,
                   path: str,
                   only_rank0: bool = False,
                   tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        self.strategy.save_model(model=self.model, path=path, only_rank0=only_rank0, tokenizer=tokenizer)