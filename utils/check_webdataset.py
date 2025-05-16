import argparse
from tqdm import tqdm

import webdataset as wds


def check_webdataset(args):
    dataset = wds.WebDataset(args.dataset_path)

    total = 0
    for _ in tqdm(dataset):
        total += 1

    print(f"Total number of samples: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/data/bingda/journeydb/data/train/processed/journeydb-train-{000..199}.tar")
    args = parser.parse_args()

    check_webdataset(args)