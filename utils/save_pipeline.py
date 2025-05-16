import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from io import BytesIO
import zstandard as zstd

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
import torch
from transformers import CLIPTokenizer, GemmaTokenizer, AutoModel, CLIPTextModelWithProjection

from diffusion.configs import DiTConfig, FuseDiTConfig
from diffusion.models import DiT, FuseDiT
from diffusion.pipelines import DiTPipeline, FuseDiTPipeline, FuseDiTPipelineWithCLIP


def main(args):
    if args.trainer == "deepspeed":
        state_dict = get_fp32_state_dict_from_zero_checkpoint(args.checkpoint)
    elif args.trainer == "spmd":
        if args.compression:
            with open(f"{args.checkpoint}/model.pt.zst", "rb") as f:
                compressed = f.read()
            data = zstd.decompress(compressed)
            state_dict = torch.load(BytesIO(data))
        else:
            state_dict = torch.load(f"{args.checkpoint}/model.pt")

        for k in list(state_dict.keys()):
            if "_orig_module." in k:
                state_dict[k.replace("_orig_module.", "")] = state_dict[k]
                del state_dict[k]
    else:
        raise ValueError(f"Unknown trainer: {args.trainer}")

    if args.type == "baseline-dit":
        config = DiTConfig.from_pretrained(args.checkpoint)
        transformer = DiT(config)
        transformer.load_state_dict(state_dict)
    elif "fuse-dit" in args.type:
        config = FuseDiTConfig.from_pretrained(args.checkpoint)
        transformer = FuseDiT(config)
        transformer.load_state_dict(state_dict)

    tokenizer = GemmaTokenizer.from_pretrained(config.base_config._name_or_path)

    if args.type == "baseline-dit":
        pipeline = DiTPipeline(
            transformer=transformer,
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler, subfolder="scheduler"),
            vae=AutoencoderKL.from_pretrained(args.vae, subfolder="vae"),
            tokenizer=tokenizer,
            llm=AutoModel.from_pretrained(config.base_config._name_or_path),
        )
    elif args.type == "fuse-dit":
        pipeline = FuseDiTPipeline(
            transformer=transformer,
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler, subfolder="scheduler"),
            vae=AutoencoderKL.from_pretrained(args.vae, subfolder="vae"),
            tokenizer=tokenizer
        )
    elif args.type == "fuse-dit-clip":
        pipeline = FuseDiTPipelineWithCLIP(
            transformer=transformer,
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler, subfolder="scheduler"),
            vae=AutoencoderKL.from_pretrained(args.vae, subfolder="vae"),
            tokenizer=tokenizer,
            clip=CLIPTextModelWithProjection.from_pretrained(args.clip_l, subfolder="text_encoder"),
            clip_tokenizer=CLIPTokenizer.from_pretrained(args.clip_l, subfolder="tokenizer"),
        )
    else:
        raise ValueError(f"Unknown type: {args.type}")
    
    pipeline.save_pretrained(os.path.join(args.checkpoint, "pipeline"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./output")
    parser.add_argument("--trainer", type=str, default="deepspeed")
    parser.add_argument("--type", type=str, default="fuse-dit")
    parser.add_argument("--scheduler", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--vae", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--clip_l", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--clip_g", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--t5", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--compression", action="store_true")
    args = parser.parse_args()
    main(args)