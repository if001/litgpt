# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import random_split
from datasets import concatenate_datasets, load_dataset


from litgpt import PromptStyle
from litgpt.data import Alpaca, SFTDataset

_URL: str = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"


def format(ds):
    if 'query' in ds:
        text = ds['query']
    elif 'input' in ds and 'instruction' in ds:
        text = ds['instruction'] + "\n" + ds["input"]
    else:
        text = ds['instruction']

    if 'answer' in ds:
        output = ds['answer']
    else:
        output = ds['output']
    if text is None or output is None:
        return None
    return dict({"instruction": text, "output": output})

@dataclass
class SFTDatasetHF(Alpaca):
    """Dolly data module for supervised finetuning."""

    mask_prompt: bool = False
    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    val_split_fraction: float = 0.01
    """The fraction of the dataset to use for the validation dataset. The rest is used for training."""
    prompt_style: Union[str, PromptStyle] = "alpaca"
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    ignore_index: int = -100
    """The index to use for elements to be ignored in the label."""
    seed: int = 42
    """The random seed for creating the train/val splits and shuffling the dataset."""
    num_workers: int = 4
    """How many DataLoader processes to use for loading."""
    repo_ids: str = ""


    def __post_init__(self) -> None:
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        ds_list = []
        for id in self.repo_ids.split(","):
            _ds = load_dataset(id, split="train")
            ds_list.append(_ds)

        ds = concatenate_datasets(ds_list).shuffle(seed=42).train_test_split(test_size=self.val_split_fraction)

        train_data = []
        for v in ds['train']:
            _v = format(v)
            if _v:
                train_data.append(_v)

        test_data = []
        for v in ds['test']:
            _v = format(v)
            if _v:
                test_data.append(_v)
        
        train_data, test_data = list(train_data), list(test_data)

        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = SFTDataset(
            data=test_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

