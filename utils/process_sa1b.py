import argparse
import os
import multiprocessing
from tqdm import tqdm

from PIL import Image
from torchvision.transforms.functional import resize
import webdataset as wds


def process_shard(shard_index, images, dataset_path, captions_path, output_dir):
    shard = wds.TarWriter(f"{output_dir}/sa1b-train-{shard_index:04d}.tar")
    print(f"Writing to {output_dir}/sa1b-train-{shard_index:04d}.tar")

    for sample in tqdm(images):
        try:
            shard.write(
                {
                    "__key__": sample.replace(".jpg", ""),
                    "image.jpg": resize(Image.open(os.path.join(dataset_path, sample)), 512),
                    "synthetic_caption.txt": open(os.path.join(captions_path, sample.replace(".jpg", ".txt"))).read()
                }
            )
        except Exception as e:
            print(f"Error processing {sample}: {e}")

    shard.close()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    images = os.listdir(args.dataset_path)
    print(f"Found {len(images)} images")

    with multiprocessing.Pool(64) as pool:
        pool.starmap(
            process_shard,
            [
                (shard_index, images[shard_index * args.samples_per_shard: min((shard_index + 1) * args.samples_per_shard, len(images))], args.dataset_path, args.captions_path, args.output_dir)
                for shard_index in range(len(images) // args.samples_per_shard + 1)
            ]
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/data/bingda/sa1b/images")
    parser.add_argument("--captions_path", type=str, default="/data/bingda/sa1b/captions")
    parser.add_argument("--samples_per_shard", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="/data/bingda/sa1b/processed/")
    args = parser.parse_args()

    main(args)