from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from io import BytesIO
import os
import random
import shutil
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import is_torch_xla_available, is_wandb_available
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from transformers import CLIPTextModelWithProjection
import zstandard as zstd

from .data import get_dataloader
from .models import build_model, get_llm, update_self_attention_mask

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_backend
    from torch_xla.distributed.fsdp.utils import apply_xla_patch_to_nn_linear
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.spmd as xs
    from torch_xla.experimental.distributed_checkpoint import CheckpointManager, prime_optimizer
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel
    import torch_xla.runtime as xr

    ACCEL = "xla"
elif torch.cuda.is_available():
    import deepspeed
    
    ACCEL = "cuda"
else:
    ACCEL = "cpu"

if is_wandb_available():
    import wandb


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if ACCEL == "xla":
        xm.set_rng_state(seed)


def sync_to_cpu(state_dict):
    def convert_fn(item):
        if isinstance(item, torch.Tensor):
            item = xm._maybe_convert_to_cpu(item).to(torch.float32)
            return item
        elif isinstance(item, dict):
            return {k: convert_fn(v) for k,v in item.items()}
        elif isinstance(item, list):
            return [convert_fn(v) for v in item]
        elif isinstance(item, tuple):
            return tuple(convert_fn(v) for v in item)
        else:
            return item
    state_dict = {
        k: convert_fn(v) for k,v in state_dict.items()
    }
    return state_dict


def rank_zero_only(func):
    def wrapper(*args, **kwargs):
        if ACCEL == "xla":
            xm.mark_step()
        if dist.get_rank() == 0:
            return func(*args, **kwargs)
    return wrapper


def unwrap(model):
    if ACCEL == "cuda":
        return model.module
    else:
        return model


class Trainer(ABC):
    @rank_zero_only
    def before_training(self):
        if is_wandb_available():
            wandb.init(
                project=self.hparams.trainer.project,
                name=self.hparams.trainer.run,
                config=OmegaConf.to_container(self.hparams, resolve=True)
            )
        print("***** Running training *****")
        print(f"Total train batch size = {self.total_batch_size}")
        print(f"Total optimization steps = {self.hparams.trainer.max_steps}")
        self.progress_bar = tqdm(total=self.hparams.trainer.max_steps, initial=self.global_step)

    @abstractmethod
    def backward(self, loss):
        pass

    @abstractmethod
    def optimizer_step(self):
        pass

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(dtype=dtype)
        step_indices = [(self.noise_scheduler.timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def dit_training_step(self, batch):
        pixel_values = batch["pixel_values"]

        if self.hparams.model.encoder_type == "llm":
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].to(self.device)

            llm_attention_mask = update_self_attention_mask(attention_mask, 0, False, self.device, torch.float32 if ACCEL == "xla" else self.llm.dtype)

            position_ids = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0)

            text_hidden_states = self.llm(
                input_ids.to(self.device),
                llm_attention_mask,
                position_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=False,
            )[1][self.hparams.model.text_hidden_states_index].to(dtype=pixel_values.dtype)
        else:
            raise ValueError(f"Unknown encoder type: {self.hparams.model.encoder}")

        # Convert images to latent space
        model_input = pixel_values.to(self.device)
        model_input = self.vae.encode(model_input).latent_dist.sample()
        model_input = (model_input - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input).to(self.device)
        bsz = model_input.shape[0]

        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.hparams.trainer.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.hparams.trainer.logit_mean,
            logit_std=self.hparams.trainer.logit_std,
            mode_scale=self.hparams.trainer.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices]

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype).to(self.device)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # Predict the noise residual
        output = self.model(
            hidden_states=noisy_model_input,
            timestep=timesteps.to(self.device),
            text_hidden_states=text_hidden_states,
            attention_mask=attention_mask,
        )
        model_pred = output

        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        if self.hparams.trainer.precondition_outputs:
            model_pred = model_pred * (-sigmas) + noisy_model_input

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.hparams.trainer.weighting_scheme, sigmas=sigmas)

        # flow matching loss
        if self.hparams.trainer.precondition_outputs:
            target = model_input
        else:
            target = noise - model_input

        # Compute regular loss.
        loss = torch.mean((weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1)
        loss = loss.mean()

        return loss

    def fusedit_training_step(self, batch):
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        if self.hparams.model.encoder_type == "clip-llm":
            clip_input_ids = batch["clip_input_ids"]
            text_modulation_embeds = self.clip(clip_input_ids.to(self.device)).text_embeds.to(dtype=pixel_values.dtype)
        else:
            text_modulation_embeds = None

        # Convert images to latent space
        model_input = pixel_values.to(self.device)
        model_input = self.vae.encode(model_input).latent_dist.sample()
        model_input = (model_input - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input).to(self.device)
        bsz = model_input.shape[0]

        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.hparams.trainer.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.hparams.trainer.logit_mean,
            logit_std=self.hparams.trainer.logit_std,
            mode_scale=self.hparams.trainer.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices]

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype).to(self.device)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # Predict the noise residual
        output = self.model(
            hidden_states=noisy_model_input,
            timestep=timesteps.to(self.device),
            input_ids=input_ids.to(self.device),
            text_modulation_embeds=text_modulation_embeds,
            attention_mask=attention_mask.to(self.device),
        )
        model_pred = output[0]

        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        if self.hparams.trainer.precondition_outputs:
            model_pred = model_pred * (-sigmas) + noisy_model_input

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.hparams.trainer.weighting_scheme, sigmas=sigmas)

        # flow matching loss
        if self.hparams.trainer.precondition_outputs:
            target = model_input
        else:
            target = noise - model_input

        # Compute regular loss.
        loss = torch.mean((weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1)
        loss = loss.mean()

        return loss

    @rank_zero_only
    def log(self, data, progress_bar=True):
        if is_wandb_available():
            wandb.log(data, step=self.global_step)
        if progress_bar:
            self.progress_bar.update(1)
            self.progress_bar.set_postfix(data)

    @torch.no_grad()
    def update_ema(self):
        if self.hparams.ema.update_steps is not None and self.global_step % self.hparams.ema.update_steps == 0:
            ema_params = OrderedDict(self.ema.named_parameters())
            model_params = OrderedDict(unwrap(self.model).named_parameters())
            assert set(ema_params.keys()) == set(model_params.keys())

            for name, param in [(k, v) for k, v in model_params.items() if v.requires_grad]:
                ema_params[name].mul_(self.hparams.ema.decay).add_(param.data, alpha=1 - self.hparams.ema.decay)

    @abstractmethod
    def save_checkpoint(self):
        pass

    def train(self):
        self.before_training()

        for step in range((self.hparams.trainer.max_steps - self.global_step) * self.hparams.trainer.gradient_accumulation_steps):
            with torch.autocast(ACCEL, dtype=torch.bfloat16 if self.hparams.trainer.mixed_precision == "bf16" else torch.float32):
                batch = next(self.dataloader)
                loss = self.training_step(batch)

            self.backward(loss)

            if (step + 1) % self.hparams.trainer.gradient_accumulation_steps == 0:
                self.optimizer_step()
                
                self.global_step += 1

                if self.global_step % self.hparams.trainer.logging_steps == 0:
                    self.log({ "train/loss": loss.detach().item(), "train/lr": self.lr_scheduler.get_last_lr()[0] })
                elif dist.get_rank() == 0:
                    self.progress_bar.update(1)

                self.update_ema()
                self.save_checkpoint()

        self.after_training()


class SPMDTrainer(Trainer):
    def __init__(self, hparams):
        self.hparams = hparams

        if self.hparams.trainer.seed is not None:
            seed_everything(hparams.trainer.seed)

        xr.use_spmd()
        if hparams.trainer.cache_dir is not None:
            try:
                xr.initialize_cache(hparams.trainer.cache_dir, readonly=False)
            except Exception as e:
                print(f"Failed to initialize cache: {e}")
        dist.init_process_group(backend="gloo", init_method="xla://")
        num_devices = xr.global_runtime_device_count()
        device_ids = np.arange(num_devices)
        mesh_shape = (num_devices, 1)
        mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
        xs.set_global_mesh(mesh)
        self.mesh = xs.get_global_mesh()
        self.device = xm.xla_device()
        self.total_batch_size = hparams.data.batch_size * dist.get_world_size() * hparams.trainer.gradient_accumulation_steps
        self.global_step = 0

        self.model = build_model(self.hparams)
        if self.hparams.trainer.resume_from is not None:
            state_dict = torch.load(self.hparams.trainer.resume_from)
            self.model.load_state_dict(state_dict)
        if self.hparams.ema.update_steps is not None:
            self.ema = deepcopy(self.model)

        self.model = SpmdFullyShardedDataParallel(self.model)
        self.model = apply_xla_patch_to_nn_linear(self.model, xs.xla_patched_nn_linear_forward)

        if self.hparams.ema.update_steps is not None:
            self.ema = SpmdFullyShardedDataParallel(self.ema)
            self.ema = apply_xla_patch_to_nn_linear(self.ema, xs.xla_patched_nn_linear_forward)
            self.ema.requires_grad_(False)

        self.vae = AutoencoderKL.from_pretrained(**self.hparams.vae)
        self.vae.requires_grad_(False)
        self.vae.to(self.device)

        if self.hparams.model.name == "DiT":
            if self.hparams.model.encoder_type == "llm":
                self.llm = get_llm(self.hparams.model.base, self.model.config.base_config)
                self.llm.requires_grad_(False)
                self.llm = SpmdFullyShardedDataParallel(self.llm)
                self.llm = apply_xla_patch_to_nn_linear(self.llm, xs.xla_patched_nn_linear_forward)
            else:
                raise ValueError(f"Unknown encoder type: {self.hparams.model.encoder_type}")
            self.training_step = self.dit_training_step
        elif self.hparams.model.name == "FuseDiT":
            if self.hparams.model.encoder_type == "clip-llm":
                self.clip = CLIPTextModelWithProjection.from_pretrained(**self.hparams.clip_l)
                self.clip.requires_grad_(False)
                self.clip.to(self.device)
            self.training_step = self.fusedit_training_step
        else:
            raise ValueError(f"Unknown model name: {self.hparams.model.name}")

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(**self.hparams.noise_scheduler)

        self.optimizer = torch.optim.AdamW(self.model.trainable_parameters(), **self.hparams.optimizer)
        self.lr_scheduler = get_scheduler(**self.hparams.lr_scheduler, optimizer=self.optimizer, num_training_steps=self.hparams.trainer.max_steps)
        
        self.dataloader = get_dataloader(self.hparams)
        input_sharding = {
            "pixel_values": xs.ShardingSpec(self.mesh, ("fsdp", None, None, None), minibatch=True),
        }
        if self.hparams.model.encoder_type == "llm":
            input_sharding["input_ids"] = xs.ShardingSpec(self.mesh, ("fsdp", None), minibatch=True)
            input_sharding["attention_mask"] = xs.ShardingSpec(self.mesh, ("fsdp", None), minibatch=True)
        elif self.hparams.model.encoder_type == "clip-llm":
            input_sharding["clip_input_ids"] = xs.ShardingSpec(self.mesh, ("fsdp", None), minibatch=True)
            input_sharding["input_ids"] = xs.ShardingSpec(self.mesh, ("fsdp", None), minibatch=True)
            input_sharding["attention_mask"] = xs.ShardingSpec(self.mesh, ("fsdp", None), minibatch=True)
        else:
            raise ValueError(f"Unknown encoder type: {self.hparams.model.encoder_type}")
        self.dataloader = pl.MpDeviceLoader(
            self.dataloader,
            self.device,
            input_sharding=input_sharding,
            loader_prefetch_size=self.hparams.data.loader_prefetch_size,
            device_prefetch_size=self.hparams.data.device_prefetch_size,
        )
        self.dataloader = iter(self.dataloader)

        self.checkpoint_manager = CheckpointManager(self.hparams.trainer.checkpoint_dir, self.hparams.trainer.checkpointing_steps)
        self.load_checkpoint()

    def load_checkpoint(self):
        tracked_steps = self.checkpoint_manager.all_steps()

        if tracked_steps:
            self.global_step = max(tracked_steps)

            if not self.hparams.trainer.hard_skip_resume:
                for i in tqdm(range(self.global_step * self.hparams.trainer.gradient_accumulation_steps), disable=not dist.get_rank() == 0):
                    next(self.dataloader)
                    if (i + 1) % 10000 == 0:
                        xm.rendezvous("resume")

            prime_optimizer(self.optimizer)

            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            if self.hparams.ema.update_steps is not None:
                state_dict["ema"] = self.ema.state_dict()

            self.checkpoint_manager.restore(self.global_step, state_dict)
            self.model.load_state_dict(state_dict["model"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
            if self.hparams.ema.update_steps is not None:
                self.ema.load_state_dict(state_dict["ema"])

            checkpoint_dir = os.path.join(self.hparams.trainer.checkpoint_dir, str(self.global_step))
            scheduler_state = torch.load(os.path.join(checkpoint_dir, "scheduler.ckpt"))
            self.lr_scheduler.load_state_dict(scheduler_state)

            if not self.hparams.trainer.hard_skip_resume:
                rng_state = torch.load(os.path.join(checkpoint_dir, f"rng-{dist.get_rank()}.ckpt"))
                torch.set_rng_state(rng_state["torch"])
                np.random.set_state(rng_state["numpy"])
                random.setstate(rng_state["random"])
                xm.set_rng_state(rng_state["torch_xla"])

        xm.rendezvous("start")

    def save_checkpoint(self):
        if self.global_step % self.hparams.trainer.checkpointing_steps == 0:
            xm.mark_step()

            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            if self.hparams.ema.update_steps is not None:
                state_dict["ema"] = self.ema.state_dict()
            self.checkpoint_manager.save(self.global_step, state_dict)

            checkpoint_dir = os.path.join(self.hparams.trainer.checkpoint_dir, str(self.global_step))
            rng_state = {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
                "torch_xla": xm.get_rng_state(),
            }
            torch.save(rng_state, os.path.join(checkpoint_dir, f"rng-{dist.get_rank()}.ckpt"))

            if dist.get_rank() == 0:  
                self.model.config.save_pretrained(checkpoint_dir)
                torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.ckpt"))

                last_checkpoint = self.global_step - self.hparams.trainer.checkpointing_steps
                if last_checkpoint > 0 and last_checkpoint % self.hparams.trainer.consolidation_steps != 0:
                    shutil.rmtree(os.path.join(self.hparams.trainer.checkpoint_dir, str(self.global_step - self.hparams.trainer.checkpointing_steps)))

            if self.global_step % self.hparams.trainer.consolidation_steps == 0:
                state_dict = sync_to_cpu(state_dict)
                if dist.get_rank() == 0:
                    if self.hparams.ema.update_steps is not None:
                        with BytesIO() as buffer, open(os.path.join(checkpoint_dir, "ema.pt.zst"), 'wb') as f:
                            torch.save(state_dict["ema"], buffer)
                            f.write(zstd.compress(buffer.getvalue()))
                    else:
                        with BytesIO() as buffer, open(os.path.join(checkpoint_dir, "model.pt.zst"), 'wb') as f:
                            torch.save(state_dict["model"], buffer)
                            f.write(zstd.compress(buffer.getvalue()))

            xm.rendezvous("save")

    def backward(self, loss):
        (loss / self.hparams.trainer.gradient_accumulation_steps).backward()

    def optimizer_step(self):
        torch.nn.utils.clip_grad_norm_(
            list(filter(lambda p: p.requires_grad, self.model.parameters())),
            self.hparams.trainer.gradient_clipping
        )
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

    def after_training(self):
        xm.rendezvous("end")


class DeepSpeedTrainer(Trainer):
    def __init__(self, hparams, local_rank):
        self.hparams = hparams

        if self.hparams.trainer.seed is not None:
            seed_everything(hparams.trainer.seed)

        deepspeed.init_distributed()
        deepspeed.get_accelerator().set_device(local_rank)
        self.device = torch.device(deepspeed.get_accelerator().device_name(), local_rank)
        self.total_batch_size = hparams.data.batch_size * dist.get_world_size() * hparams.trainer.gradient_accumulation_steps
        self.global_step = 0

        self.model = build_model(self.hparams)
        self.model = self.model.to(self.device)
        self.model.train()
        
        if self.hparams.trainer.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.hparams.ema.update_steps is not None:
            self.ema = deepcopy(self.model)
            self.ema.requires_grad_(False)

        self.vae = AutoencoderKL.from_pretrained(**self.hparams.vae)
        self.vae.requires_grad_(False)
        self.vae = self.vae.to(self.device)

        if self.hparams.model.name == "DiT":
            if self.hparams.model.encoder_type == "llm":
                self.llm = get_llm(self.hparams.model.base, self.model.config.base_config)
                self.llm.requires_grad_(False)
                self.llm.to(self.device)
            else:
                raise ValueError(f"Unknown encoder type: {self.hparams.model.encoder_type}")
            self.training_step = self.dit_training_step
        elif self.hparams.model.name == "FuseDiT":
            if self.hparams.model.encoder_type == "clip-llm":
                self.clip = CLIPTextModelWithProjection.from_pretrained(**self.hparams.clip_l)
                self.clip.requires_grad_(False)
                self.clip.to(self.device)
            self.training_step = self.fusedit_training_step
        else:
            raise ValueError(f"Unknown model name: {self.hparams.model.name}")

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(**self.hparams.noise_scheduler)

        self.optimizer = torch.optim.AdamW(self.model.trainable_parameters(), **self.hparams.optimizer)
        self.lr_scheduler = get_scheduler(**self.hparams.lr_scheduler, optimizer=self.optimizer, num_training_steps=self.hparams.trainer.max_steps)

        self.dataloader = get_dataloader(self.hparams)
        self.dataloader = iter(self.dataloader)

        config = {
            "train_micro_batch_size_per_gpu": self.hparams.data.batch_size,
            "gradient_accumulation_steps": self.hparams.trainer.gradient_accumulation_steps,
            "gradient_clipping": self.hparams.trainer.gradient_clipping,
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": 2,
            },
            "flops_profiler": {
                "enabled": True,
                "output_file": os.path.join(self.hparams.trainer.checkpoint_dir, "flops.txt"),
            }
        }
        self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            config=config,
        )

        self.load_checkpoint()

    def load_checkpoint(self):
        if self.hparams.trainer.resume_from is not None:
            if self.hparams.trainer.resume_from == "latest":
                if os.path.exists(os.path.join(self.hparams.trainer.checkpoint_dir, "latest")):
                    with open(os.path.join(self.hparams.trainer.checkpoint_dir, "latest")) as f:
                        self.hparams.trainer.resume_from = int(f.read())
                else:
                    return

            _, client_state = self.model.load_checkpoint(self.hparams.trainer.checkpoint_dir, str(self.hparams.trainer.resume_from))
            self.global_step = int(self.hparams.trainer.resume_from)

            if not self.hparams.trainer.hard_skip_resume:
                for _ in tqdm(range(self.global_step * self.hparams.trainer.gradient_accumulation_steps), disable=not dist.get_rank() == 0):
                    next(self.dataloader)

            if self.hparams.ema.update_steps is not None:
                self.ema.load_state_dict(client_state["ema"])

            if not self.hparams.trainer.hard_skip_resume:
                rng_state = torch.load(os.path.join(self.hparams.trainer.checkpoint_dir, str(self.global_step), f"rng-{dist.get_rank()}.ckpt"))
                torch.set_rng_state(rng_state["torch"])
                np.random.set_state(rng_state["numpy"])
                random.setstate(rng_state["random"])
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

    def save_checkpoint(self):
        if self.global_step % self.hparams.trainer.checkpointing_steps == 0:
            unwrap(self.model).config.save_pretrained(self.hparams.trainer.checkpoint_dir)

            self.model.save_checkpoint(
                self.hparams.trainer.checkpoint_dir,
                str(self.global_step),
                { "ema": self.ema.state_dict() } if self.hparams.ema.update_steps is not None else {}
            )

            rng_state = {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            }
            torch.save(rng_state, os.path.join(self.hparams.trainer.checkpoint_dir, str(self.global_step), f"rng-{dist.get_rank()}.ckpt"))

            if self.global_step % self.hparams.trainer.consolidation_steps == 0:
                if dist.get_rank() == 0:
                    if self.hparams.ema.update_steps is not None:
                        with BytesIO() as buffer, open(os.path.join(self.hparams.trainer.checkpoint_dir, str(self.global_step), "ema.pt.zst"), 'wb') as f:
                            torch.save(self.ema.state_dict(), buffer)
                            f.write(zstd.compress(buffer.getvalue()))
                    else:
                        with BytesIO() as buffer, open(os.path.join(self.hparams.trainer.checkpoint_dir, str(self.global_step), "model.pt.zst"), 'wb') as f:
                            torch.save(unwrap(self.model).state_dict(), buffer)
                            f.write(zstd.compress(buffer.getvalue()))

    def backward(self, loss):
        self.model.backward(loss)

    def optimizer_step(self):
        self.model.step()

    def after_training(self):
        pass


def get_trainer(hparams, local_rank):
    if ACCEL == "xla":
        return SPMDTrainer(hparams)
    elif ACCEL == "cuda":
        return DeepSpeedTrainer(hparams, local_rank)
    else:
        raise NotImplementedError