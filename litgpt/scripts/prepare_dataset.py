from datasets import load_dataset
import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--key', default='text')
    args = parser.parse_args()
    return args


def main():
    args = parse()
    print('load', args.repo_id)
    if 'json' in args.dataset_path:
        ds = load_dataset('json', data_files=args.repo_id, split="train")
    else:
        ds = load_dataset(args.repo_id, split="train")    
    print(ds)

    texts = []
    for v in ds:
        texts.append(v[args.key]+"\n")
    
    with open(args.output_file, 'w') as f:
        f.writelines(texts)
    print('end...', args.output_file)

if __name__ == '__main__':
    main()