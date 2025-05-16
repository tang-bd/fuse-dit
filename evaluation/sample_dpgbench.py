import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from accelerate import PartialState
import fire
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import torch

from diffusion.pipelines import DiTPipeline, FuseDiTPipeline, FuseDiTPipelineWithCLIP


def load_pipeline(model_type: str, ckpt_path: str):
    if model_type == "baseline-dit":
        pipeline = DiTPipeline.from_pretrained(ckpt_path).to("cuda")
    elif model_type == "fuse-dit":
        pipeline = FuseDiTPipeline.from_pretrained(ckpt_path).to("cuda")
    elif model_type == "fuse-dit-clip":
        pipeline = FuseDiTPipelineWithCLIP.from_pretrained(ckpt_path).to("cuda")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


@torch.no_grad()
def generate(opt):
    
    prompts = os.listdir(opt.gen.prompts_dir)
    
    distributed_state = PartialState()
    with distributed_state.split_between_processes(prompts) as samples:
        for model in tqdm(opt.pipeline.ckpt_path):
            os.makedirs(os.path.join(model, f"dpgbench-{int(opt.gen.scale)}"), exist_ok=True)
            pipe = load_pipeline(opt.pipeline.model_type, os.path.join(model, "pipeline"))
            for sample in tqdm(samples):
                prompt = open(os.path.join(opt.gen.prompts_dir, sample)).read()

                generator = torch.manual_seed(opt.gen.seed)
                with torch.autocast("cuda"):
                    images = pipe(
                        prompt=opt.gen.instruction + prompt,
                        height=opt.gen.H,
                        width=opt.gen.W,
                        num_inference_steps=opt.gen.steps,
                        guidance_scale=opt.gen.scale,
                        num_images_per_prompt=4,
                        negative_prompt=opt.gen.negative_prompt or None,
                        generator=generator,
                        instruction=opt.gen.instruction,
                    )[0]
                
                grid_image = Image.new('RGB', (opt.gen.W * 2, opt.gen.H * 2))
                for i, image in enumerate(images):
                    grid_image.paste(image, (i % 2 * opt.gen.W, i // 2 * opt.gen.H))
                grid_image.save(os.path.join(model, "dpgbench-6", f"{os.path.splitext(sample)[0]}.png"))


def main(config_file):
    hparams = OmegaConf.load(config_file)
    generate(hparams)


if __name__ == "__main__":
    fire.Fire(main)