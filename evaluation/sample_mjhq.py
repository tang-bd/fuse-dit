import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json

import torch
from tqdm import tqdm

from accelerate import PartialState
from lightning import seed_everything

from diffusion.pipelines import DiTPipeline, FuseDiTPipeline, FuseDiTPipelineWithCLIP


def sample(args, data_dict):
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    distributed_state = PartialState()
    if args.model_type == "baseline-dit":
        pipeline = DiTPipeline.from_pretrained(args.checkpoint).to("cuda")
    elif args.model_type == "fuse-dit":
        pipeline = FuseDiTPipeline.from_pretrained(args.checkpoint).to("cuda", dtype=torch.bfloat16)
    elif args.model_type == "fuse-dit-clip":
        pipeline = FuseDiTPipelineWithCLIP.from_pretrained(args.checkpoint).to("cuda", dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    pipeline.set_progress_bar_config(disable=True)

    

    with distributed_state.split_between_processes(list(data_dict.items())[:args.num_samples]) as samples:
        samples = [tuple(zip(*samples[i:i + args.batch_size])) for i in range(0, len(samples), args.batch_size)]
        for file_names, info in tqdm(samples):
            categories = [sample["category"] for sample in info]
            captions = [sample["prompt"] for sample in info]
            with torch.autocast("cuda"):
                images = pipeline(
                    captions,
                    width=args.resolution,
                    height=args.resolution,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    use_cache=True
                )[0]
            for category, file_name, image in zip(categories, file_names, images):
                os.makedirs(os.path.join(args.save_dir, category), exist_ok=True)
                image.save(os.path.join(args.save_dir, category, file_name + ".jpg"))


def main(args):
    with open(args.prompts) as f:
        data_dict = json.load(f)

    sample(args, data_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/data/bingda/ckpts/large-scale-800k/pipeline")
    parser.add_argument("--model_type", type=str, default="fuse-dit")
    parser.add_argument("--prompts", type=str, default="/data/bingda/mjhq/captions.json")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="/data/bingda/ckpts/large-scale-800k/mjhq-6")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)