# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional, Union

import torch
from torch.utils.data import DataLoader, random_split
from datasets import concatenate_datasets, load_dataset
from datasets import Dataset

from litgpt import Tokenizer
from litgpt.data import Alpaca
from litgpt.data.packed_dataset import prepare_packed_dataloader
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling

def format(ds):
    if 'query' in ds and 'answer':
        text = ds['query']
        output = ds['answer']
    
    if 'instruction' in ds and 'output' in ds:
        if 'input' in ds:
            text = ds['instruction'] + "\n" + ds["input"]
        else:
            text = ds['instruction']
        output = ds['output']

    if 'question_ja' in ds and 'generated_solution_ja' in ds:
        text = ds['question_ja']
        output = ds['generated_solution_ja']

    if ('q1' in ds) and ('a1' in ds) and ('q2' in ds) and ('a2' in ds):
        text = ds['q1'] + "\n" \
        + "### アシスタント:\n" \
        + ds['a1'] + "\n" \
        + "### ユーザー:\n" \
        + ds['q2']  + "\n\n"
        output = ds['a2']

    if text is None or output is None:
        return None
    prompt = f"""### ユーザー:
{text}

### アシスタント:
{output}
"""
    return dict({"text": prompt})

@dataclass
class SFTPackedDatasetHF(Alpaca):
    val_split_fraction: float = 0.01
    """The fraction of the dataset to use for the validation dataset. The rest is used for training."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""
    repo_ids: str = ""

    # num_of_sequences: int = 1024
    num_of_sequences: int = 5
    chars_per_token: int = 2
    append_concat_token: bool = True
    add_special_tokens: bool = True
    dataset_text_field: str = "text"

    def connect(
        self, tokenizer_id: str = "", batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length


    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        json_list = []
        for id in self.repo_ids.split(","):
            _ds = load_dataset(id, split="train")
            for v in _ds:
                _v = format(v)
                if _v:
                    json_list.append(_v)

        _dataset = Dataset.from_list(json_list)
        _dataset = _dataset.shuffle(seed=self.seed).train_test_split(test_size=self.val_split_fraction)
        self.train_dataset = _dataset['train']
        print('before train dataset', self.train_dataset)
        self.test_dataset = _dataset['test']

    
    def train_dataloader(self) -> DataLoader:
        dataset = prepare_packed_dataloader(
                self.tokenizer,
                self.train_dataset,
                self.dataset_text_field,
                self.max_seq_length,
                self.num_of_sequences,
                self.chars_per_token,
                None,
                self.append_concat_token,
                self.add_special_tokens,
            )
        return DataLoader(dataset, batch_size=self.batch_size, 
                          shuffle=False, pin_memory=True, collate_fn=self.data_collator)

    def val_dataloader(self) -> DataLoader:
        dataset = prepare_packed_dataloader(
                self.tokenizer,
                self.test_dataset,
                self.dataset_text_field,
                self.max_seq_length,
                self.num_of_sequences,
                self.chars_per_token,
                None,
                self.append_concat_token,
                self.add_special_tokens,
            )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, 
                          pin_memory=True, collate_fn=self.data_collator)
