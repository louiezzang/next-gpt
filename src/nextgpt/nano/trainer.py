"""
Trainer for GPT model.

@author Younggue Bae
"""
import os
import sys
import time
import math
import pickle
from tqdm import tqdm
import dataclasses
from contextlib import nullcontext

import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch import nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from .model import GPTConfig, GPT


class GPTTrainer(object):
    def __init__(self, args, dataloader, device, init_from="scratch", checkpoint=None, logger=None):
        """
        Constructor.

        Args:
            args: The arguments
            dataloader: The dataloader
            device: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
            init_from: The model initialization option ('scratch', 'resume', 'pretrained' or 'gpt2...')
            checkpoint: The checkpoint dict when resumes training from a checkpoint
            logger: The model logger which is `ModelLogger` object
        """
        self.args = args
        print(f"args: {vars(args)}")
        self.dataloader = dataloader
        self.init_from = init_from
        self.checkpoint = checkpoint
        self.logger = logger
        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks if hasattr(args, "metric_ks") else [10]
        self.best_metric = args.best_metric if hasattr(args, "best_metric") else None
        # Evaluation
        self.generative_function = args.generative_function if hasattr(args, "generative_function") else "generate"
        self.temperature = args.temperature if hasattr(args, "temperature") else 1.0
        self.top_k = args.top_k if hasattr(args, "top_k") else None

        # Fix random seed
        model_init_seed = args.model_init_seed if hasattr(args, "model_init_seed") else 0
        fix_random_seed_as(model_init_seed)

        # Data
        self.gradient_accumulation_steps = 5  # used to simulate larger batch sizes

        # Model
        self.vocab_size = args.vocab_size if hasattr(args, "vocab_size") else None
        self.block_size = self.args.block_size if hasattr(args, "block_size") else 1024
        self.model, self.model_args = self.__init_model()

        # Init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        self.start_epoch_num = 0
        self.best_val_loss = 1e9 

        # Print the number of parameters in the model.
        # print(f"number of parameters: {sum(p.numel() for p in self.model.parameters())/1e6}M")

        # Adamw optimizer
        self.lr = self.args.lr if hasattr(args, "lr") else 6e-4
        weight_decay = self.args.weight_decay if hasattr(args, "weight_decay") else 1e-1
        beta1 = self.args.beta1 if hasattr(args, "beta1") else 0.9
        beta2 = self.args.beta2 if hasattr(args, "beta2") else 0.95
        self.grad_clip = self.args.grad_clip if hasattr(args, "grad_clip") else 1.0 # clip gradients at this value, or disable if == 0.0

        # Learning rate decay settings
        self.decay_lr = self.args.decay_lr if hasattr(args, "decay_lr") else True # whether to decay the learning rate
        self.warmup_epochs = self.args.warmup_epochs if hasattr(args, "warmup_epochs") else 2000 # how many steps to warm up for
        self.lr_decay_epochs = self.args.lr_decay_epochs if hasattr(args, "lr_decay_epochs") else 600000 # should be ~= max_iters per Chinchilla
        self.min_lr = self.args.min_lr if hasattr(args, "min_lr") else 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

        # System
        self.dtype = self.args.dtype if hasattr(args, "dtype") else "bfloat16" # "float32", "bfloat16", or "float16", the latter will auto implement a GradScaler
        self.compile = self.args.compile if hasattr(args, "compile") else False # use PyTorch 2.0 to compile the model to be faster
        self.print_interval = self.args.print_interval if hasattr(args, "print_interval") else 1
        self.log_interval = self.args.log_interval if hasattr(args, "log_interval") else 1
        self.eval_interval = self.args.eval_interval if hasattr(args, "eval_interval") else 1
        self.eval_iters = self.args.eval_iters if hasattr(args, "eval_iters") else 200
        self.always_save_checkpoint = self.args.always_save_checkpoint if hasattr(args, "always_save_checkpoint") else True
        
        # Initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == "float16"))

        # Optimizer
        device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.model.to(device)
        self.optimizer = self.model.configure_optimizers(weight_decay, self.lr, (beta1, beta2), device_type=device_type)
        if init_from == "resume":
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Compile the model.
        if self.compile:
            print("compiling the model... (takes a ~minute)")
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model) # requires PyTorch 2.0
    
    def __init_model(self):
        # Model init.
        all_args = vars(self.args) # Convert Namespace to dict
        model_args = dict()
        for field in dataclasses.fields(GPTConfig):
            field_name = field.name
            if field_name in all_args:
                model_args[field_name] = all_args[field_name]

        if self.init_from == "scratch":
            # Init a new model from scratch
            print("Initializing a new model from scratch")
            # Determine the vocab size we'll use for from-scratch training
            if self.vocab_size is None:
                print("Defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
                self.vocab_size = 50304
                self.args.vocab_size = 50304
            
            model_args["vocab_size"] = self.vocab_size
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
        elif self.init_from == "resume":
            print(f"Resuming training from a checkpoint")
            # Resume training from a checkpoint.
            checkpoint = self.checkpoint
            checkpoint_model_args = checkpoint["model_args"]
            # Force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line.
            for k in [field.name for field in dataclasses.fields(GPTConfig)]:
                model_args[k] = checkpoint_model_args[k]
            
            # Create the model.
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            state_dict = checkpoint["model_state_dict"]
            # Fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = "_orig_mod."
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            self.start_epoch_num = checkpoint["epoch"]
            self.best_val_loss = checkpoint["best_val_loss"]
        elif self.init_from == "pretrained":
            print(f"Finetuning from a pretrained checkpoint")
            # Finetune from a pretrained checkpoint.
            checkpoint = self.checkpoint
            # checkpoint_model_args = checkpoint["model_args"]

            # Force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line.
            # for k in [field.name for field in dataclasses.fields(GPTConfig)]:
            #     model_args[k] = checkpoint_model_args[k]
            
            # Create the model.
            model_args["vocab_size"] = self.vocab_size
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            state_dict = checkpoint["model_state_dict"]
            # Fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = "_orig_mod."
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
        elif self.init_from.startswith("gpt2"):
            print(f"Initializing from OpenAI GPT-2 weights: {self.init_from}")
            # Initialize from OpenAI GPT-2 weights.
            override_args = dict(dropout=0.0)
            model = GPT.from_pretrained(self.init_from, override_args)
            # Read off the created config params, so we can store them into checkpoint correctly.
            for k in [field.name for field in dataclasses.fields(GPTConfig)]:
                model_args[k] = getattr(model.config, k)

        # Crop down the model block size if desired, using model surgery.
        if self.block_size < model.config.block_size:
            model.crop_block_size(self.block_size)
            model_args["block_size"] = self.block_size # So that the checkpoint will have the right value

        return model, model_args
    
    def get_lr(self):
        """ Returns the learning rate of the current optimizer.
        """
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]
        
    # learning rate decay scheduler (cosine with warmup)
    def get_epoch_lr(self, epoch):
        """ Gets the learning rate decay scheduler (cosine with warmup).
        """

        # 1) linear warmup for warmup_epochs steps
        if epoch < self.warmup_epochs:
            return self.lr * epoch / self.warmup_epochs
        # 2) if epoch > lr_decay_epochs, return min learning rate
        if epoch > self.lr_decay_epochs:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (epoch - self.warmup_epochs) / (self.lr_decay_epochs - self.warmup_epochs)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.min_lr + coeff * (self.lr - self.min_lr)

    @staticmethod
    def init_process(rank, size, backend="nccl"):
        """
        Initializes the distributed environment.

        Args:
            rank: The device id
            size: The number of GPUs
            backend: gloo, nccl
        """
        print(f"init_process: rank= {rank}, backend = {backend}")
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL" # set to DETAIL for runtime logging.
        dist.init_process_group(backend=backend, rank=rank, world_size=size)

    @staticmethod
    def cleanup_process():
        """
        Cleanups and terminates the training process on GPUs.
        """
        try:
            dist.destroy_process_group()
        except Exception as err:
            print(err)
            os.system('kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk "{print $2}") ')

        print("=== Destroyed torch.multiprocessing group! ===")

    def run(self, n_gpus, distributed=False, distributed_backend="nccl", enable_tqdm=False):
        """
        Runs the training process.

        Args:
            n_gpus: The number of GPUs
            distributed: Will utilize multi-GPUs on PyTorch DDP(DistributedDataParallel) if true
            enable_tqdm: Will show tqdm progress bar if true
            distributed_backend: PyTorch distributed backend (gloo, mpi, nccl)
        """
        if distributed:
            try:
                mp.spawn(self.train,
                         args=(n_gpus, distributed, distributed_backend, enable_tqdm), # arguments must be an iterable
                         nprocs=n_gpus,
                         join=True)
            except (InterruptedError, KeyboardInterrupt, Exception) as err:
                print(err)
                self.cleanup_process()
                raise err
        else:
            if n_gpus == 0: # CPU mode
                self.train(rank=0, world_size=0, distributed=distributed, distributed_backend=distributed_backend, enable_tqdm=enable_tqdm)
            else: # GPU mode
                # The `world_size` should be 1 to prevent DistributedSampler from distributing dataset.
                self.train(rank=0, world_size=1, distributed=distributed, distributed_backend=distributed_backend, enable_tqdm=enable_tqdm)

    def train(self, rank, world_size, distributed=False, distributed_backend="nccl", enable_tqdm=False):
        if distributed:
            self.init_process(rank, world_size, distributed_backend)
            device = f"cuda:{rank}"
            torch.cuda.set_device(device) # no need this?
            seed_offset = rank # Each process gets a different seed
        else:
            if world_size > 0:
                device = "cuda"
            else:
                device = "cpu"
            seed_offset = 0
            self.gradient_accumulation_steps *= world_size # simulate n gpus (eg. 8)

        torch.manual_seed(1337 + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        device_type = "cuda" if world_size > 0 else "cpu" # for later use in torch.autocast

        print(f"device: {device}, device_type: {device_type}")

        # note: float16 data type will automatically use a GradScaler
        ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[self.dtype]
        ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # Get data loader.
        train_loader, val_loader, test_loader = self.dataloader.get_pytorch_dataloaders(rank, world_size)

        self.model = self.model.to(device)

        if distributed:
            # self.model = DistributedDataParallel(self.model, device_ids=[rank], find_unused_parameters=True)
            self.model = DistributedDataParallel(self.model, device_ids=[rank])

        self.t0 = time.time()
        self.running_mfu = -1.0  # model flops(floating point operations per second) utilization (MFU)
        for epoch in range(self.start_epoch_num, self.num_epochs):
            self.train_one_epoch(epoch, rank, world_size, device, train_loader, distributed, ctx, enable_tqdm=enable_tqdm)
            if val_loader is not None:
                self.validate(epoch, rank, world_size, device, val_loader, distributed, ctx, enable_tqdm=enable_tqdm)

 
    def train_one_epoch(self, epoch, rank, world_size, device, dataloader, distributed, ctx, enable_tqdm=False):
        self.model.train()
        raw_model = self.model.module if distributed else self.model # Unwrap DDP container if needed

        # Determine and set the learning rate for this iteration.
        lr = self.get_epoch_lr(epoch) if self.decay_lr else self.lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(dataloader, file=sys.stdout) if enable_tqdm else None

        total_batch_size = len(dataloader)
        print_per_every = total_batch_size // self.print_interval
        for batch_idx, batch in enumerate(tqdm_dataloader if enable_tqdm else dataloader):
            if len(batch) == 3:
                X, Y, _ = batch
            else:    
                X, Y = batch
            batch_size = Y.shape[0]

            if device == "cpu":
                X, Y = X.to(device), Y.to(device)
            else:
                # Pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                X, Y = X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(device, non_blocking=True)

            # Forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(self.gradient_accumulation_steps):
                if distributed:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable.
                    self.model.require_backward_grad_sync = (micro_step == self.gradient_accumulation_steps - 1)
                with ctx:
                    logits, loss = self.model(X, Y)

                # Backward pass, with gradient scaling if training in fp16.
                self.scaler.scale(loss).backward()

            # Clip the gradient.
            if self.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            # Step the optimizer and scaler if training in fp16.
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # Flush the gradients as soon as we can, no need for this memory anymore.
            self.optimizer.zero_grad(set_to_none=True)

            lossf = loss.item() # loss as float. note: this is a CPU-GPU sync point
            average_meter_set.update("train_loss", lossf)

            # timing and logging
            t1 = time.time()
            dt = t1 - self.t0
            self.t0 = t1
            if rank == 0:
                mfu = raw_model.estimate_mfu(batch_size * self.gradient_accumulation_steps, dt)
                self.running_mfu = mfu if self.running_mfu == -1.0 else 0.9*self.running_mfu + 0.1*mfu
                progress = ((batch_idx + 1) / total_batch_size) * 100
                if enable_tqdm:
                    description = "Epoch {}, train: loss {:.4f}, lr {}, mfu {:.2f}% ".format(
                        epoch, average_meter_set["train_loss"].avg, self.get_lr(), self.running_mfu*100)
                    tqdm_dataloader.set_description(description)
                elif (print_per_every > 0 and batch_idx % print_per_every == 0) or batch_idx == total_batch_size - 1:
                    description = "Epoch {}, train: loss {:.4f}, lr {} mfu {:.2f}%:  {:.0f}% | {:d}/{:d}".format(
                        epoch, average_meter_set["train_loss"].avg, self.get_lr(),  self.running_mfu*100, progress, batch_idx+1, total_batch_size)
                    print(description)

        # Save checkpoint.
        if self.logger and rank == 0:
            self.logger.log_metrics(average_meter_set.averages(), epoch)

    @torch.no_grad()
    def validate(self, epoch, rank, world_size, device, dataloader, distributed, ctx, enable_tqdm=False):
        self.model.eval()
        raw_model = self.model.module if distributed else self.model # Unwrap DDP container if needed

        average_meter_set = AverageMeterSet()

        total_batch_size = len(dataloader)
        print_per_every = total_batch_size // self.print_interval

        tqdm_dataloader = tqdm(dataloader) if enable_tqdm else None
        for batch_idx, batch in enumerate(tqdm_dataloader if enable_tqdm else dataloader):
            Z = None
            if len(batch) == 3:
                X, Y, Z = batch
            else:    
                X, Y = batch

            batch_size = Y.shape[0]

            if device == "cpu":
                X, Y = X.to(device), Y.to(device)
                if Z is not None:
                    Z = Z.to(device)
            else:
                # Pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                X, Y = X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(device, non_blocking=True)
                if Z is not None:
                    Z = Z.pin_memory().to(device, non_blocking=True)

            # print(f"X.shape = {X.shape}, Y.shape = {Y.shape}, Z.shape = {Z.shape if Z is not None else None}")

            with ctx:
                logits, loss = self.model(X, Y)
                lossf = loss.item() # loss as float. note: this is a CPU-GPU sync point
                average_meter_set.update("val_loss", lossf)
                if Z is not None:
                    if self.generative_function == "topk":
                        pred = raw_model.topk(X, top_k=50, temperature=self.temperature)
                    else:
                        pred = raw_model.generate(X, 50, temperature=self.temperature, top_k=self.top_k, no_repeat=True, with_input=False)
                    # print(f"*** pred: {pred.shape}")
                    metrics = self.calculate_metrics(pred, Z)
                    for k, v in metrics.items():
                        average_meter_set.update(k, v)
                    
            if rank == 0:
                progress = ((batch_idx + 1) / total_batch_size) * 100
                metric_keys = (
                    ["val_loss"] +
                    ["NDCG@%d" % k for k in self.metric_ks[:3]] +
                    ["Recall@%d" % k for k in self.metric_ks[:3]]
                )

                metric_description = ", ".join([f"{k} {average_meter_set[k].avg:.4f}" for k in metric_keys if k in average_meter_set.meters])
                
                if enable_tqdm:
                    description = "Epoch {}, val: {}".format(epoch, metric_description)
                    tqdm_dataloader.set_description(description)
                elif (print_per_every > 0 and batch_idx % print_per_every == 0) or batch_idx == total_batch_size - 1:
                    description = "Epoch {}, val: {}:  {:.0f}% | {:d}/{:d}".format(
                        epoch,
                        metric_description,
                        progress,
                        batch_idx + 1,
                        total_batch_size)
                    print(description)

        if self.logger and rank == 0:
            self.logger.log_metrics(average_meter_set.averages(), epoch)
            if average_meter_set["val_loss"].avg < self.best_val_loss or self.always_save_checkpoint:
                self.best_val_loss = average_meter_set["val_loss"].avg
                if epoch >= 0:
                    state_dict = {
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "model_args": vars(self.args),
                        "epoch": epoch,
                        "best_val_loss": self.best_val_loss,
                    }
                    self.logger.save_checkpoint(epoch=epoch, state_dict=state_dict)

    def calculate_metrics(self, pred, labels):
        metrics = recalls_and_ndcgs_for_ks(pred, labels, self.metric_ks)
        return metrics


class GPTRandomSampleTrainer(GPTTrainer):
    """
    Random sample trainer.
    """

    def train(self, rank, world_size, distributed=False, distributed_backend="nccl", enable_tqdm=False):
        if distributed:
            self.init_process(rank, world_size, distributed_backend)
            device = f"cuda:{rank}"
            torch.cuda.set_device(device) # no need this?
            seed_offset = rank # Each process gets a different seed
        else:
            if world_size > 0:
                device = "cuda"
            else:
                device = "cpu"
            seed_offset = 0
            self.gradient_accumulation_steps *= world_size # simulate n gpus (eg. 8)

        torch.manual_seed(1337 + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        device_type = "cuda" if world_size > 0 else "cpu" # for later use in torch.autocast

        print(f"device: {device}, device_type: {device_type}")

        # note: float16 data type will automatically use a GradScaler
        ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[self.dtype]
        ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        dataloader = self.dataloader # Note that this is GPTRandomSampleDataloader!

        self.model = self.model.to(device)

        if distributed:
            # self.model = DistributedDataParallel(self.model, device_ids=[rank], find_unused_parameters=True)
            self.model = DistributedDataParallel(self.model, device_ids=[rank])

        raw_model = self.model.module if distributed else self.model # Unwrap DDP container if needed

        t0 = time.time()
        running_mfu = -1.0  # model flops(floating point operations per second) utilization (MFU)
        epochs = range(self.start_epoch_num, self.num_epochs)
        tqdm_dataloader = tqdm(epochs, file=sys.stdout) if enable_tqdm else None
        for epoch in tqdm_dataloader if enable_tqdm else epochs:
            train_loss = self.train_one_epoch(epoch, rank, world_size, device, dataloader, distributed, ctx)
            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if epoch % self.log_interval == 0 and rank == 0:
                train_lossf = train_loss.item() # loss as float. note: this is a CPU-GPU sync point
                if epoch >= 5: # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(dataloader.batch_size * self.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                description = f"Epoch {epoch}: loss {train_lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
                if enable_tqdm:
                    tqdm_dataloader.set_description(description)
                else:
                    print(description)
                if self.logger:
                    self.logger.log_metrics({"train_loss": train_lossf}, epoch)
            
            # Evaluate the loss on train/val sets and write checkpoints.
            if epoch % self.eval_interval == 0 and rank == 0:
                eval_losses = self.validate(epoch, rank, world_size, device, dataloader, distributed, ctx)
                description = f"Epoch {epoch}: train loss {eval_losses['train']:.4f}, val loss {eval_losses['val']:.4f}"
                if enable_tqdm:
                    tqdm_dataloader.set_description(description)
                else:
                    print(description)

                if eval_losses["val"] < self.best_val_loss or self.always_save_checkpoint:
                    self.best_val_loss = eval_losses["val"]
                    if epoch > 0 and self.logger:
                        state_dict = {
                            "model_state_dict": raw_model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "model_args": vars(self.args),
                            "epoch": epoch,
                            "best_val_loss": self.best_val_loss,
                        }
                        print(f"saving checkpoint: epoch={epoch}, best_val_loss={self.best_val_loss}")
                        self.logger.save_checkpoint(epoch=epoch, state_dict=state_dict)
                if self.logger:
                    self.logger.log_metrics({"val_loss": eval_losses["val"].item()}, epoch)
            
    def train_one_epoch(self, epoch, rank, world_size, device, dataloader, distributed, ctx):
        self.model.train()

        # Determine and set the learning rate for this iteration.
        lr = self.get_epoch_lr(epoch) if self.decay_lr else self.lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        X, Y = dataloader.get_batch("train")
        batch_size = Y.shape[0]
        
        if device == "cpu":
            X, Y = X.to(device), Y.to(device)
        else:
            # Pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            X, Y = X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(device, non_blocking=True)

        # Forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(self.gradient_accumulation_steps):
            if distributed:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable.
                self.model.require_backward_grad_sync = (micro_step == self.gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = self.model(X, Y)

            # Backward pass, with gradient scaling if training in fp16.
            self.scaler.scale(loss).backward()

        # Clip the gradient.
        if self.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        # Step the optimizer and scaler if training in fp16.
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # Flush the gradients as soon as we can, no need for this memory anymore.
        self.optimizer.zero_grad(set_to_none=True)

        return loss

    @torch.no_grad()
    def validate(self, epoch, rank, world_size, device, dataloader, distributed, ctx):
        self.model.eval()
 
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = dataloader.get_batch(split)

                if device == "cpu":
                    X, Y = X.to(device), Y.to(device)
                else:
                    # Pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                    X, Y = X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(device, non_blocking=True)

                with ctx:
                    logits, loss = self.model(X, Y)

                losses[k] = loss.item()
                out[split] = losses.mean()

        return out


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string="{}"):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string="{}"):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string="{}"):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string="{}"):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def recalls_and_ndcgs_for_ks(pred, labels, ks):
    metrics = {}

    pred = pred.cpu()
    labels = labels.cpu()
    answer_count = labels.sum(1)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    cut = pred

    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        # print(f"*** cut = {cut.shape}")
        hits = labels_float.gather(1, cut)

        recall = hits.sum(1) / answer_count_float
        metrics[f"Recall@{k}"] = recall[~torch.isnan(recall)].mean().item()

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights).sum(1)
        idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count])
        ndcg = dcg / idcg
        ndcg = ndcg[~torch.isnan(ndcg)].mean()
        metrics[f"NDCG@{k}"] = ndcg.item()

    return metrics