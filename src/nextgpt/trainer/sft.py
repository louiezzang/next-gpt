import math
import time
from abc import ABC
from typing import Optional, List

import loralib as lora
import torch
import torch.distributed as dist
# import wandb
from ..models.loss import GPTLMLoss
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
# from transformers.trainer import get_scheduler # This may cause pyarrow version issue due to the dependency of datasets lib!!!
from transformers.optimization import get_scheduler

from .callbacks import Callback
from .strategies import Strategy
from .utils import is_rank_0


class SFTTrainer(ABC):
    """
        Trainer to use while training reward model.
    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataloader: the dataloader to use for training
        eval_dataloader: the dataloader to use for evaluation
        # batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        accumulation_steps (int, defaults to 8): the number of accumulation steps
        # optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
        lr_scheduler_type (str, defaults to cosine): the scheduler type to use (linear, cosine)
    """

    def __init__(
        self,
        model,
        strategy: Strategy,
        optim: Optimizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader = None,
        # batch_size: int = 1,
        max_epochs: int = 2,
        accumulation_steps: int = 8,
        lr_scheduler_type: str = "cosine",
        callbacks: List[Callback] = [],
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.callbacks = callbacks

        self.model = strategy.setup_model(model)
        if "DDP" in str(self.strategy):
            self.model = self.model.module
        self.optimizer = strategy.setup_optimizer(optim, self.model)

        self.accumulation_steps = accumulation_steps
        num_update_steps_per_epoch = len(train_dataloader) // self.accumulation_steps
        max_steps = math.ceil(self.epochs * num_update_steps_per_epoch)

        self.scheduler = get_scheduler(lr_scheduler_type,
                                       self.optimizer,
                                       num_warmup_steps=math.ceil(max_steps * 0.03),
                                       num_training_steps=max_steps)

    def fit(self, logger=None, log_interval=10, verbose=False):
        # wandb.init(project="nextGPT", name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # wandb.watch(self.model)
        total_loss = 0
        # epoch_bar = tqdm(range(self.epochs), desc='Epochs', disable=not is_rank_0())
        step_bar = tqdm(range(len(self.train_dataloader) // self.accumulation_steps * self.epochs),
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

                loss = loss / self.accumulation_steps

                self.strategy.backward(loss, self.model, self.optimizer)

                total_loss += loss.item()

                # gradient accumulation
                if (batch_id + 1) % self.accumulation_steps == 0:
                    self.strategy.optimizer_step(self.optimizer)
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    
                    if is_rank_0():
                        global_step = (batch_id + 1) + (epoch * len(self.train_dataloader))
                        self._on_log_metrics(
                            metrics={"train_loss": total_loss / self.accumulation_steps},
                            step=global_step
                        )
                    
                    step_bar.update()
                    step_bar.set_postfix({
                        "loss": total_loss / self.accumulation_steps,
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