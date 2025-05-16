import argparse
import os
import multiprocessing
from tqdm import tqdm

import duckdb
import webdataset as wds


def process_shard(shard_index, dataset_path, output_dir):
    dataset = wds.WebDataset(dataset_path.format(shard_index))
    shard = wds.TarWriter(f"{output_dir}/cc12m-train-{shard_index:04d}.tar")
    print(f"Writing to {output_dir}/cc12m-train-{shard_index:04d}.tar")

    for sample in tqdm(dataset):
        try:
            synthetic_caption = duckdb.sql(
                f"SELECT caption_llava FROM captions WHERE key='{sample['__key__']}'"
            ).fetchall()[0][0]

            shard.write(
                {
                    "__key__": sample["__key__"],
                    "image.jpg": sample["jpg"],
                    "original_caption.txt": sample["txt"],
                    "synthetic_caption.txt": synthetic_caption
                }
            )
        except Exception as e:
            print(e)

    shard.close()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    duckdb.sql(f"CREATE TABLE captions AS SELECT * FROM read_json('{args.captions_path}');")

    with multiprocessing.Pool() as pool:
        pool.starmap(
            process_shard,
            [(shard_index, args.dataset_path, args.output_dir) for shard_index in range(args.num_shards)]
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/data/bingda/cc12m-wds/cc12m-train-{:04d}.tar")
    parser.add_argument("--num_shards", type=int, default=2176)
    parser.add_argument("--captions_path", type=str, default="/data/bingda/cc12m-captions/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="/data/bingda/cc12m-recaptioned")
    args = parser.parse_args()

    main(args)