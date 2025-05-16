import argparse
from omegaconf import OmegaConf

from diffusion.trainers import get_trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1) # only used for DeepSpeed
    args = parser.parse_args()

    hparams = OmegaConf.load(args.config)
    trainer = get_trainer(hparams, args.local_rank)
    trainer.train()