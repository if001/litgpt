# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import os
import time
from pathlib import Path

from litgpt import Tokenizer
from litgpt.data.prepare_starcoder import DataChunkRecipe
from litgpt.utils import CLI


class JsonlDataRecipe(DataChunkRecipe):
    def __init__(self, tokenizer: Tokenizer, chunk_size: int):
        super().__init__(chunk_size)
        self.tokenizer = tokenizer
        self.is_generator = True
        self.total_tokens = 0

    def prepare_structure(self, input_dir):
        files = Path(input_dir).rglob("*.jsonl")
        return [str(file) for file in files]

    def prepare_item(self, filepath):
        from datasets import load_dataset
        ds = load_dataset('json', data_files=filepath, split="train")
        for v in ds:
            text = v['text']
            text_ids = self.tokenizer.encode(text, bos=False, eos=True)
            self.total_tokens += len(text_ids)
            yield text_ids

def prepare(
    dataset_ids: str = "",
    tmp_dir: Path = Path("/tmp/train_datasets"),
    output_dir: Path = Path("data/slimpajama/train"),
    tokenizer_path: Path = Path("checkpoints/Llama-2-7b-hf/"),
    chunk_size: int = (2049 * 16384),
    fast_dev_run: bool = False,
    dataset_max_len: int = -1,
    use_cache: bool = False
) -> None:
    assert dataset_ids != ""
    dataset_ids = dataset_ids.split(',')
    from datasets import load_dataset
    for id in dataset_ids:
        if use_cache == True:
            print('use cache...')
            continue
        ds = load_dataset(id, split="train")
        tmp_name = id.split('/')[-1]
        if dataset_max_len != -1 and len(ds) > dataset_max_len:
            print('dataset splited...', len(ds), int(len(ds)/dataset_max_len))
            for idx, num in enumerate(range(0, len(ds), dataset_max_len)):
                name = str(tmp_dir /f'{tmp_name}_{idx}.jsonl')
                end = (num+dataset_max_len) if len(ds) > num+dataset_max_len else len(ds)
                _tmp = ds.select(range(num, end))
                _tmp.to_json(name, force_ascii=False)
        else:
            ds.to_json(str(tmp_dir /f'{tmp_name}.jsonl'), force_ascii=False)

    from litdata.processing.data_processor import DataProcessor

    tokenizer = Tokenizer(tokenizer_path)
    data_recipe = JsonlDataRecipe(tokenizer=tokenizer, chunk_size=chunk_size)
    data_processor = DataProcessor(
        input_dir=str(tmp_dir),
        output_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        num_workers=os.cpu_count(),
        num_downloaders=1,
    )

    start_time = time.time()
    data_processor.run(data_recipe)
    elapsed_time = time.time() - start_time
    print(f"total tokens:", data_recipe.total_tokens)
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    CLI(prepare)