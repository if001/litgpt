# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import glob
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from torch.utils.data import DataLoader

from litgpt import Tokenizer
from litgpt.data import DataModule


@dataclass
class HFData(DataModule):
    """The TextFile data module used for pretraining.

    Reads in text data from plaintext files contained in a data folder
    and provides training and validation dataloaders that return batches of tokens.
    Every sample is set to a fixed length.
    """
    train_data_path: Path
    """The path to the data directory used for training that contains .txt files"""
    val_data_path: Optional[Path] = None
    """The path to the data directory used for validation that
    contains .txt files. Splits off data for validation from the
    training set if None."""
    seed: int = 42
    """The seed to use for shuffling the dataset."""
    num_workers: int = 4
    """The number of workers to use for data loading."""

    out_path_train: str = ""
    out_path_val: str = ""

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        assert self.out_path_train != ""
        assert self.out_path_val != ""
        os.makedirs(self.out_path_train, exist_ok=True)
        os.makedirs(self.out_path_val, exist_ok=True)
        self.out_path_train = Path(self.out_path_train)
        self.out_path_val = Path(self.out_path_val)

    def connect(self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: int = -1) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def prepare_data(self) -> None:
        from litdata import optimize

        train_files = self.train_data_path.split(",")
        val_files = self.val_data_path.split(",")
        # It's ok to use almost all CPUs here because this runs in a single process
        num_workers = os.cpu_count() - 1
        # num_workers = 1
        use_workers = min(num_workers, len(train_files))
        if not Path(self.out_path_train).is_dir():
            validate_tokenizer(self.tokenizer)
            optimize(
                fn=partial(tokenize, tokenizer=self.tokenizer),
                inputs=train_files,
                output_dir=str(self.out_path_train),
                num_workers=use_workers,
                chunk_bytes="50MB",
            )
        use_workers = min(num_workers, len(val_files))
        if not Path(self.out_path_val).is_dir():
            validate_tokenizer(self.tokenizer)
            optimize(
                fn=partial(tokenize, tokenizer=self.tokenizer),
                inputs=val_files,
                output_dir=str(self.out_path_val),
                num_workers=use_workers,
                chunk_bytes="50MB",
            )

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader
        train_dataset = StreamingDataset(
            input_dir=str(self.out_path_train),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=True,
            drop_last=True,
        )
        train_dataloader = StreamingDataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataset, TokensLoader

        val_dataset = StreamingDataset(
            input_dir=str(self.out_path_val),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=True,
            # Consider setting to False, but we would lose some samples due to truncation when world size > 1
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return val_dataloader


def tokenize(filename: str, tokenizer: Tokenizer):
    from datasets import load_dataset
    ds = load_dataset(filename, split="train")
    for v in ds:
        yield tokenizer.encode(v['text'], bos=True, eos=True)

def validate_tokenizer(tokenizer: Tokenizer) -> None:
    if tokenizer is None:
        raise ValueError(
            "Tokenizer is None. If you are using this data module via `litgpt pretrain`, "
            "please provide a valid `--tokenizer_dir` path."
        )
