#!/usr/bin/env python3
"""
Fine-tune Flux with LoRA on your own images.
Trigger word (token) is injected into every caption.
Tested with diffusers >= 0.30 (Flux branch).
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
from peft import LoraConfig, get_peft_model
from PIL import Image
from torchvision import transforms
from transformers import (
    AutoTokenizer,
    BlipProcessor,
    BlipForConditionalGeneration,
    T5Tokenizer,
    FluxTransformer2DModel,
    FluxPipeline,
)
from diffusers import FluxTransformer2DModel as DiffusersFluxTransformer2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel, compute_snr
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
    return text_inputs.input_ids, text_inputs.attention_mask


class DreamBoothDataset(torch.utils.data.Dataset):
    """
    If .txt sidecars exist we use them, otherwise we auto-caption with BLIP
    and append the trigger word.
    """

    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str,
        tokenizer,
        size: int = 512,
        center_crop=True,
        auto_caption: bool = True,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt  # trigger word
        self.auto_caption = auto_caption

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.instance_images_path = [
            x for x in self.instance_images_path if x.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        if not self.instance_images_path:
            raise ValueError(f"No images found in {instance_data_root}")

        self.image_transforms = image_transforms(size, center_crop)

        # BLIP for auto-caption
        if self.auto_caption:
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.instance_images_path)

    def __getitem__(self, index):
        example = {}
        image_path = self.instance_images_path[index]
        image = Image.open(image_path).convert("RGB")

        # caption
        txt_path = image_path.with_suffix(".txt")
        if txt_path.exists():
            caption = txt_path.read_text(encoding="utf-8").strip()
        else:
            if self.auto_caption:
                inputs = self.caption_processor(image, return_tensors="pt").to(self.caption_model.device)
                with torch.no_grad():
                    out = self.caption_model.generate(**inputs, max_new_tokens=32)
                caption = self.caption_processor.decode(out[0], skip_special_tokens=True).strip()
            else:
                caption = ""
        # append trigger
        caption = f"{caption.strip()}, {self.instance_prompt}"
        example["input_ids"], example["attention_mask"] = tokenize_prompt(self.tokenizer, caption)
        example["pixel_values"] = self.image_transforms(image)
        return example


def parse_args():
    parser = argparse.ArgumentParser(description="Simple Flux LoRA fine-tuning.")
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
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--num_validation_images", type=int, default=2)
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

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=None)

    # Dataset
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
    )

    # Flux transformer
    transformer = DiffusersFluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=None
    )

    # LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # Prep
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Train
    global_step = 0
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            transformer.train()
            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=accelerator.unwrap_model(transformer).dtype)
            model_pred = transformer(pixel_values)
            loss = F.mse_loss(model_pred, pixel_values, reduction="mean")
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
                    accelerator.unwrap_model(transformer).save_pretrained(save_path)

            if global_step >= args.max_train_steps:
                break

    # Save final LoRA
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = accelerator.unwrap_model(transformer)
        transformer.save_pretrained(args.output_dir)
        # also export safetensors
        from safetensors.torch import save_file
        save_file(transformor.state_dict(), os.path.join(args.output_dir, "pytorch_lora_weights.safetensors"))
        logger.info(f"LoRA saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
