from datasets import load_dataset
import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--repo_id', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse()
    print('load', args.dataset_path)
    ds = load_dataset('json', data_files=args.dataset_path, split="train")
    print(ds)
    ds.push_to_hub(args.repo_id)

if __name__ == '__main__':
    main()