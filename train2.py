#!/usr/bin/env python3
"""
QLoRA fine-tune Flux-dev on your own images.
Trigger word (token) is injected into every caption.
~9 GB VRAM on a 16 GB card.
"""

import argparse
import math
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import HfFolder
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from diffusers import (
    FluxTransformer2DModel,
    FluxPipeline,
    FluxScheduler,
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import check_min_version, deprecate

check_min_version("0.30.0.dev0")
logger = get_logger(__name__)


def image_transforms(size: int, center_crop=True):
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Flux expects [-1,1]
        ]
    )


def tokenize_prompt(tokenizer, prompt, max_length=77):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_inputs.input_ids


class DreamBoothDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str,
        tokenizer,
        size: int = 512,
        center_crop=True,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.instance_images_path = [
            x for x in self.instance_images_path if x.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        if not self.instance_images_path:
            raise ValueError(f"No images found in {instance_data_root}")

        self.image_transforms = image_transforms(size, center_crop)

    def __len__(self):
        return len(self.instance_images_path)

    def __getitem__(self, index):
        image_path = self.instance_images_path[index]
        image = Image.open(image_path).convert("RGB")

        # caption
        txt_path = image_path.with_suffix(".txt")
        if txt_path.exists():
            caption = txt_path.read_text(encoding="utf-8").strip()
        else:
            caption = "a photo of mamaplugxs"
        caption = f"{caption.strip()}, {self.instance_prompt}"

        example = {}
        example["input_ids"] = tokenize_prompt(self.tokenizer, caption).squeeze(0)
        example["pixel_values"] = self.image_transforms(image)
        return example


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Flux-dev fine-tuning.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="flux-lora")
    parser.add_argument("--instance_prompt", type=str, default="mamaplugxs")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_config=ProjectConfiguration(project_dir=args.output_dir),
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

    # tokenizers
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")

    # dataset
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
    )

    # 4-bit transformer
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        load_in_4bit=True,
        bnb_4bit_compute_dtype=weight_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        torch_dtype=weight_dtype,
        device_map="auto",
    )

    # QLoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # text encoders & vae & scheduler (frozen)
    pipe = FluxPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=weight_dtype)
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    vae = pipe.vae
    noise_scheduler = pipe.scheduler
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    # prepare
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # encode prompt helper
    def encode_prompt(input_ids):
        t5_out = text_encoder(input_ids.to(text_encoder.device), return_dict=False)[0]
        clip_out = text_encoder_2(input_ids.to(text_encoder_2.device), return_dict=False)[0]
        pooled = clip_out[0]
        return t5_out, pooled

    # train
    global_step = 0
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            transformer.train()
            pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

            # 1. encode to latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # 2. noise & timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 3. text embeds
            prompt_embeds, pooled_prompt_embeds = encode_prompt(batch["input_ids"])

            # 4. predict noise
            model_pred = transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            # 5. loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(transformer.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    transformer.save_pretrained(save_path)

            if global_step >= args.max_train_steps:
                break

    # final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = accelerator.unwrap_model(transformer)
        transformer.save_pretrained(args.output_dir)
        # safetensors too
        from safetensors.torch import save_file
        save_file(transformer.state_dict(), os.path.join(args.output_dir, "pytorch_lora_weights.safetensors"))
        logger.info(f"LoRA saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
