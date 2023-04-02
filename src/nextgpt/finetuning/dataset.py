"""
PyTorch Dataset for chatGPT.

@author Younggue
"""
import os
import copy
from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field

import numpy as np

import torch
from torch.utils.data import Dataset
import transformers


IGNORE_INDEX = -100


class SupervisedDataset(Dataset):
    """
    Dataset for Supervised Fine Tuning.
    """

    def __init__(self, 
                 data: Sequence[Dict], 
                 tokenizer, 
                 prompt_template,
                 prompt_fields=["prompt"], 
                 completion_field="completion",
                 verbose=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        # prompt_template example:
        '''
        (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n\n"
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{prompt}\n\n### Response:"
        )
        '''
        self.verbose = verbose

        ############################################################
        sources = []
        for example in data:
            prompt_input = prompt_template.format_map(example)
            sources.append(prompt_input)

        targets = []
        for example in data:
            completion = example.get(completion_field, "")
            targets.append(f"{completion}{tokenizer.eos_token}")

        if verbose:
            idx = 0
            print((sources[idx]))
            print((targets[idx]))
            print("Tokenizing inputs... This may take some time...")

        ############################################################
        examples = [s + t for s, t in zip(sources, targets)]

        # source data tokenized.
        sources_tokenized = self._tokenize_fn(sources, tokenizer)  # source only
        examples_tokenized = self._tokenize_fn(examples, tokenizer)  # source + target


        ## Input is source and output is source+target, but training only uses target part.
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX  # source is filled with -100.

        data_dict = dict(input_ids=input_ids, labels=labels)        
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        print("Loading data done!!: %d"%(len(self.labels)))   


    def _tokenize_fn(self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        """
        Tokenizes a list of strings.
        """
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )
        
        
    def __len__(self):
        return len(self.input_ids)

    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    Collate examples for supervised fine-tuning.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
