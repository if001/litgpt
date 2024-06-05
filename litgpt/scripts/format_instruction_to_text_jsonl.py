import os
from datasets import load_dataset
import argparse

def format(instruction= "", input = None, output = ""):
    if input:
        return f"""以下は、タスクを説明する命令と、さらなるコンテキストを提供する入力の組み合わせです。要求を適切に満たすような応答を書きなさい。
### 指示:
{instruction}

### 入力:
{input}

### 応答:
{output}
"""
    return f"""以下は、タスクを説明する指示です。要求を適切に完了させる回答を書きなさい。
### 指示:
{instruction}

### 応答:
{output}
"""

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--ins_key', default='instruction')
    parser.add_argument('--inp_key', default='input')
    parser.add_argument('--out_key', default='output')

    args = parser.parse_args()
    return args


def main():
    args = parse()
    dirname = os.path.dirname(args.output_file)
    os.makedirs(dirname, exist_ok=True)

    print('load', args.dataset_path)
    if 'json' in args.dataset_path:
        ds = load_dataset('json', data_files=args.dataset_path, split="train")
    else:
        ds = load_dataset(args.dataset_path, split="train")
    print(ds)
    
    texts = []
    for v in ds:
        if args.inp_key in v:
            text = format(instruction=v[args.ins_key], input=v[args.inp_key], output=v[args.output_key])
        else:
            text = format(instruction=v[args.ins_key], output=v[args.output_key])
        new_text = ''
        for v in text.split('\n'):
            if len(v) > 20:
                new_text += v + '\n'
        texts.append(new_text)

    with open(args.output_file, 'w') as f:
        f.writelines(texts)
    print('end...', args.output_file)

if __name__ == '__main__':
    main()