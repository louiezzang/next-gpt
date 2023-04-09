import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import is_rank_0


class PromptDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 dataset: Sequence[Dict], 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 prompt_template: str = None,
                 max_datasets_size: int = None):
        super(PromptDataset, self).__init__()
        self.prompts = []
        print("Loading data...")

        if max_datasets_size is not None:
            print(f"Limiting dataset to {max_datasets_size} examples.")
            dataset = dataset[:max_datasets_size]

        for data_dict in dataset:
            if prompt_template is None:
                prompt_text = data_dict["instruction"]
            else:
                prompt_text = prompt_template.format_map(data_dict)
            token = tokenizer(prompt_text,
                              return_tensors='pt',
                              max_length=96,
                              padding='max_length',
                              truncation=True)
            for idx in token['input_ids']:
                self.prompts.append(idx.to(torch.cuda.current_device()))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.prompts[i]
