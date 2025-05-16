import argparse
import json
import os
import multiprocessing
from tqdm import tqdm

from PIL import Image
from torchvision.transforms.functional import resize
import webdataset as wds


def process_shard(shard_index, dataset_path, output_dir):
    with open(os.path.join(output_dir, f"{shard_index:03d}.json")) as f:
        dataset = json.load(f)
    shard = wds.TarWriter(f"{output_dir}/journeydb-train-{shard_index:03d}.tar")
    print(f"Writing to {output_dir}/journeydb-train-{shard_index:03d}.tar")

    for sample in tqdm(dataset):
        try:
            shard.write(
                {
                    "__key__": os.path.splitext(sample["image"])[0],
                    "image.jpg": resize(Image.open(os.path.join(dataset_path, sample["image"])), 512),
                    "synthetic_caption.txt": sample["caption"]
                }
            )
        except Exception as e:
            print(e)

    shard.close()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    with multiprocessing.Pool(64) as pool:
        pool.starmap(
            process_shard,
            [(shard_index, args.dataset_path, args.output_dir) for shard_index in range(args.num_shards)]
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/data/bingda/journeydb/data/train/imgs")
    parser.add_argument("--num_shards", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="/data/bingda/journeydb/data/train/processed")
    args = parser.parse_args()

    main(args)