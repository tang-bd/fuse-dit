import argparse
import json
import os
from tqdm import tqdm


def main(args):
    with open(args.input_path) as f:
        captions = [json.loads(line) for line in f]

    cleaned_captions = [[] for _ in range(200)]
    for caption in tqdm(captions):
        cleaned_captions[int(caption["img_path"][2:5])].append(
            {
                "image": caption["img_path"],
                "caption": caption["prompt"]
            }
        )

    for i, shard in enumerate(cleaned_captions):
        with open(os.path.join(args.output_dir, f"{i:03d}.json"), "w") as f:
            json.dump(shard, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/data/bingda/journeydb/data/train/train_anno_realease_repath.jsonl")
    parser.add_argument("--output_dir", type=str, default="/data/bingda/journeydb/data/train/processed/")
    args = parser.parse_args()

    main(args)