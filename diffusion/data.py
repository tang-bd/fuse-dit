from functools import partial
import random

from diffusers.utils import is_torch_xla_available
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torchvision import transforms
from transformers import CLIPTokenizer, GemmaTokenizer
import webdataset as wds


def llm_preprocess_fn(hparams, tokenizer, sample):
    image = sample[hparams.data.image_column]

    if hparams.data.original_caption_rate > 0 and sample.get(hparams.data.caption_column.original) is not None and random.random() < hparams.data.original_caption_rate:
        caption = sample[hparams.data.caption_column.original]
    else:
        caption = sample[hparams.data.caption_column.synthetic]

    transform = transforms.Compose(
        [
            transforms.Resize(hparams.data.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            (transforms.CenterCrop(hparams.data.resolution) if hparams.data.center_crop else transforms.RandomCrop(hparams.data.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    pixel_values = transform(image)

    if random.random() < hparams.data.random_dropping_rate: # Randomly drop the caption
        caption = ""
    else:
        caption = hparams.data.instruction + caption

    if hparams.data.apply_chat_template and caption != "":
        tokenized = tokenizer.apply_chat_template(
            [{ "role": "user", "content": caption }],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=hparams.data.max_prompt_length + hparams.data.instruction_length,
            add_generation_prompt=hparams.data.add_generation_prompt,
            return_dict=True,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
    else:
        tokenized = tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=hparams.data.max_prompt_length + hparams.data.instruction_length,
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def llm_collate_fn(examples):
    pixel_values = [example["pixel_values"] for example in examples]
    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    input_ids = [example["input_ids"] for example in examples]
    input_ids = torch.cat(input_ids).to(memory_format=torch.contiguous_format)
    attention_mask = [example["attention_mask"] for example in examples]
    attention_mask = torch.cat(attention_mask).to(memory_format=torch.contiguous_format)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def get_llm_dataloader(hparams, *args, **kwargs):
    tokenizer = GemmaTokenizer.from_pretrained(hparams.data.tokenizer)

    if hparams.data.apply_chat_template:
        hparams.data.instruction_length = tokenizer.apply_chat_template(
            [{ "role": "user", "content": hparams.data.instruction.rstrip() }],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=hparams.data.max_prompt_length,
            add_generation_prompt=hparams.data.add_generation_prompt,
            return_dict=True,
        )["input_ids"].shape[1] - 1
    else:
        hparams.data.instruction_length = tokenizer(
            hparams.data.instruction.rstrip(),
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=hparams.data.max_prompt_length,
        ).input_ids.shape[1] - 1

    dataset = (
        wds.WebDataset(wds.ResampledShards(hparams.data.data_path, deterministic=True))
            .shuffle(1000, rng=random.Random(hparams.trainer.seed))
            .decode("pil")
            .map(
                partial(
                    llm_preprocess_fn,
                    hparams,
                    tokenizer,
                ),
            )
    )

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=hparams.data.batch_size,
        collate_fn=llm_collate_fn,
        num_workers=hparams.data.dataloader_num_workers,
        generator=torch.manual_seed(hparams.trainer.seed),
        worker_init_fn=seed_worker,
        pin_memory=False if is_torch_xla_available() else True,
        prefetch_factor=8,
    )


def clip_llm_preprocess_fn(hparams, clip_tokenizer, tokenizer, sample):
    image = sample[hparams.data.image_column]

    if hparams.data.original_caption_rate > 0 and sample.get(hparams.data.caption_column.original) is not None and random.random() < hparams.data.original_caption_rate:
        caption = sample[hparams.data.caption_column.original]
    else:
        caption = sample[hparams.data.caption_column.synthetic]

    transform = transforms.Compose(
        [
            transforms.Resize(hparams.data.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            (transforms.CenterCrop(hparams.data.resolution) if hparams.data.center_crop else transforms.RandomCrop(hparams.data.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    pixel_values = transform(image)

    if random.random() < hparams.data.random_dropping_rate: # Randomly drop the caption
        caption = ""
    else:
        caption = hparams.data.instruction + caption

    clip_input_ids = clip_tokenizer(
        caption,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77,
    ).input_ids

    if hparams.data.apply_chat_template and caption != "":
        tokenized = tokenizer.apply_chat_template(
            [{ "role": "user", "content": caption }],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=hparams.data.max_prompt_length + hparams.data.instruction_length,
            add_generation_prompt=hparams.data.add_generation_prompt,
            return_dict=True,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
    else:
        tokenized = tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=hparams.data.max_prompt_length + hparams.data.instruction_length,
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

    return {
        "pixel_values": pixel_values,
        "clip_input_ids": clip_input_ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def clip_llm_collate_fn(examples):
    pixel_values = [example["pixel_values"] for example in examples]
    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    clip_input_ids = [example["clip_input_ids"] for example in examples]
    clip_input_ids = torch.cat(clip_input_ids).to(memory_format=torch.contiguous_format)
    input_ids = [example["input_ids"] for example in examples]
    input_ids = torch.cat(input_ids).to(memory_format=torch.contiguous_format)
    attention_mask = [example["attention_mask"] for example in examples]
    attention_mask = torch.cat(attention_mask).to(memory_format=torch.contiguous_format)

    return {
        "pixel_values": pixel_values,
        "clip_input_ids": clip_input_ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def get_clip_llm_dataloader(hparams, *args, **kwargs):
    clip_tokenizer = CLIPTokenizer.from_pretrained(**hparams.data.tokenizer.clip)
    tokenizer = GemmaTokenizer.from_pretrained(hparams.data.tokenizer.llm)

    hparams.data.instruction_length = tokenizer(
        hparams.data.instruction.rstrip(),
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=hparams.data.max_prompt_length,
    ).input_ids.shape[1] - 1

    dataset = (
        wds.WebDataset(wds.ResampledShards(hparams.data.data_path, deterministic=True))
            .shuffle(1000, rng=random.Random(hparams.trainer.seed))
            .decode("pil")
            .map(
                partial(
                    clip_llm_preprocess_fn,
                    hparams,
                    clip_tokenizer,
                    tokenizer,
                ),
            )
    )

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=hparams.data.batch_size,
        collate_fn=clip_llm_collate_fn,
        num_workers=hparams.data.dataloader_num_workers,
        generator=torch.manual_seed(hparams.trainer.seed),
        worker_init_fn=seed_worker,
        pin_memory=False if is_torch_xla_available() else True,
        prefetch_factor=8,
    )


def get_dataloader(hparams, *args, **kwargs):
    if hparams.model.encoder_type == "clip-llm":
        return get_clip_llm_dataloader(hparams, *args, **kwargs)
    elif hparams.model.encoder_type == "llm":
        return get_llm_dataloader(hparams, *args, **kwargs)
    else:
        raise ValueError(f"Invalid encoder_type: {hparams.model.encoder_type}")