import math
import time
from abc import ABC
from typing import Optional, List
from tqdm import tqdm
# import wandb

import loralib as lora
import torch
import torch.distributed as dist

from torch import nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
# from transformers.trainer import get_scheduler # This may cause pyarrow version issue due to the dependency of datasets lib!!!
from transformers.optimization import get_scheduler

from ..models.loss import GPTLMLoss
from .callbacks import Callback
from .strategies import Strategy
from .utils import is_rank_0


class SFTTrainer(ABC):
    """
        Trainer to use while training reward model.
    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        data_collator (DataCollator): the data collator
        train_dataset: the dataset to use for training
        eval_dataset: the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        gradient_accumulation_steps (int, defaults to 8): the number of updates steps to accumulate the gradients for, before performing a backward/update pass
        lr (float, defaults to 5e-5): the learning rate
        lr_scheduler_type (str, defaults to linear): the scheduler type to use ('linear', 'cosine')
        callbacks: a list of callbacks to customize the training loop.
    """

    def __init__(
        self,
        model,
        strategy: Strategy,
        data_collator,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        batch_size: int = 1,
        max_epochs: int = 2,
        gradient_accumulation_steps: int = 8,
        lr: float = 5e-5,
        lr_scheduler_type: str = "linear",
        callbacks: List[Callback] = [],
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.lr = lr
        self.callbacks = callbacks

        self.train_dataloader = self.get_train_dataloader(dataset=train_dataset, batch_size=batch_size, data_collator=data_collator)
        self.eval_dataloader = self.get_eval_dataloader(dataset=eval_dataset, batch_size=batch_size, data_collator=data_collator)

        self.model = strategy.setup_model(model)
        if "DDP" in str(self.strategy):
            self.model = self.model.module

        optim = self.get_optimizer()
        self.optimizer = strategy.setup_optimizer(optim, self.model)

        self.gradient_accumulation_steps = gradient_accumulation_steps
        num_update_steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps
        max_steps = math.ceil(self.epochs * num_update_steps_per_epoch)

        self.scheduler = get_scheduler(lr_scheduler_type,
                                       self.optimizer,
                                       num_warmup_steps=math.ceil(max_steps * 0.03),
                                       num_training_steps=max_steps)
        
    def get_train_dataloader(self, dataset: Dataset, batch_size: int, data_collator) -> DataLoader:
        if "DDP" in str(self.strategy) and dist.is_initialized() and dist.get_world_size() > 1:
            print("DDP mode")
            train_sampler = DistributedSampler(dataset,
                                               shuffle=True,
                                               seed=42,
                                               drop_last=True,
                                               rank=dist.get_rank(),
                                               num_replicas=dist.get_world_size())
        else:
            train_sampler = None

        train_dataloader = DataLoader(dataset,
                                      shuffle=(train_sampler is None),
                                      sampler=train_sampler,
                                      batch_size=batch_size,
                                      collate_fn=data_collator,
                                      pin_memory=True)

        return train_dataloader

    def get_eval_dataloader(self, dataset: Optional[Dataset], batch_size: int, data_collator) -> DataLoader:
        if "DDP" in str(self.strategy) and dist.is_initialized() and dist.get_world_size() > 1:
            if dataset is not None:
                eval_sampler = DistributedSampler(dataset,
                                                  shuffle=False,
                                                  seed=42,
                                                  drop_last=False,
                                                  rank=dist.get_rank(),
                                                  num_replicas=dist.get_world_size())
        else:
            eval_sampler = None
        
        if dataset is not None:
            eval_dataloader = DataLoader(dataset,
                                         shuffle=(eval_sampler is None),
                                         sampler=eval_sampler,
                                         batch_size=batch_size,
                                         collate_fn=data_collator,
                                         pin_memory=True)
        else:
            eval_dataloader = None

        return eval_dataloader
    
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

    def fit(self, logger=None, log_interval=10, verbose=False):
        # wandb.init(project="nextGPT", name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # wandb.watch(self.model)
        total_loss = 0
        # epoch_bar = tqdm(range(self.epochs), desc='Epochs', disable=not is_rank_0())
        step_bar = tqdm(range(len(self.train_dataloader) // self.gradient_accumulation_steps * self.epochs),
                        desc=f'steps',
                        disable=not is_rank_0())
        for epoch in range(self.epochs):

            # process_bar = tqdm(range(len(self.train_dataloader)), desc=f'Train process for{epoch}', disable=not is_rank_0())
            # train
            self.model.train()
            for batch_id, batch in enumerate(self.train_dataloader):

                prompt_ids = batch["input_ids"].to(torch.cuda.current_device())
                p_mask = batch["attention_mask"].to(torch.cuda.current_device())
                labels = batch["labels"].to(torch.cuda.current_device())
                # prompt_ids = prompt_ids.squeeze(1).cuda()
                # p_mask = p_mask.squeeze(1).cuda()
                # prompt_logits = self.model(prompt_ids, attention_mask=p_mask, labels=labels)

                outputs = self.model(prompt_ids, attention_mask=p_mask, labels=labels)

                loss = outputs.loss
                prompt_logits = outputs.logits

                if loss >= 2.5:
                    # logger.warning(f"batch_id:{batch_id}, abnormal loss: {loss}")
                    if verbose:
                        print(f"batch_id:{batch_id}, abnormal loss: {loss}")

                loss = loss / self.gradient_accumulation_steps

                self.strategy.backward(loss, self.model, self.optimizer)

                total_loss += loss.item()

                # gradient accumulation
                if (batch_id + 1) % self.gradient_accumulation_steps == 0:
                    self.strategy.optimizer_step(self.optimizer)
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    
                    if is_rank_0():
                        global_step = (batch_id + 1) + (epoch * len(self.train_dataloader))
                        self._on_log_metrics(
                            metrics={"train_loss": total_loss / self.gradient_accumulation_steps},
                            step=global_step
                        )
                    
                    step_bar.update()
                    step_bar.set_postfix({
                        "loss": total_loss / self.gradient_accumulation_steps,
                        "lr": self.scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "batch_id": batch_id
                    })
                    total_loss = 0
                    # step_bar.update()

                # if batch_id % log_interval == 0:
                #     logger.info(f'Train Epoch {epoch}/{self.epochs} Batch {batch_id} Rank {dist.get_rank()} loss {loss.item()}')

                # process_bar.update()

            # eval
            if self.eval_dataloader is not None:
                self.model.eval()
                with torch.no_grad():
                    loss_sum = 0
                    num_seen = 0
                    for batch in self.eval_dataloader:
                        prompt_ids = batch["input_ids"].to(torch.cuda.current_device())
                        p_mask = batch["attention_mask"].to(torch.cuda.current_device())
                        labels = batch["labels"].to(torch.cuda.current_device())
                        # prompt_ids = prompt_ids.squeeze(1).cuda()
                        # p_mask = p_mask.squeeze(1).cuda()

                        outputs = self.model(prompt_ids, attention_mask=p_mask, labels=labels)
                        loss = outputs.loss
                        # prompt_logits = outputs.logits

                        loss_sum += loss.item()
                        num_seen += prompt_ids.size(0)

                    loss_mean = loss_sum / num_seen
                    step_bar.update()
                    if is_rank_0():
                        # logger.info(f'Eval Epoch {epoch}/{self.epochs} loss {loss_mean}')
                        step_bar.set_postfix({'epoch': epoch, 'eval_loss': loss_mean})

            # epoch_bar.update()

    def save_model(self,
                   path: str,
                   only_rank0: bool = False,
                   tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        self.strategy.save_model(model=self.model, path=path, only_rank0=only_rank0, tokenizer=tokenizer)

    def _on_log_metrics(self, metrics: dict, **kwargs) -> None:
        for callback in self.callbacks:
            if hasattr(callback, "on_log_metrics"):
                callback.on_log_metrics(metrics, **kwargs)