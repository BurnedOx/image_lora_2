#!/usr/bin/env python3
"""
QLoRA fine-tune Flux-dev on your own images.
Trigger word (token) is injected into every caption.
Keeps EVERYTHING in fp16 to avoid HalfTensor vs FloatTensor mismatch.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from diffusers import FluxPipeline

logger = get_logger(__name__)


def image_transforms(size: int, center_crop=True):
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # to [-1, 1]
        ]
    )


def tokenize_prompt(tokenizer, prompt, max_length=77):
    return tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.squeeze(0)


class DreamBoothDataset(torch.utils.data.Dataset):
    def __init__(self, instance_data_root, instance_prompt, tokenizer, size=512):
        self.size = size
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.image_transforms = image_transforms(size)

        self.instance_images_path = [
            p for p in Path(instance_data_root).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ]
        if not self.instance_images_path:
            raise ValueError(f"No images found in {instance_data_root}")

    def __len__(self):
        return len(self.instance_images_path)

    def __getitem__(self, index):
        image_path = self.instance_images_path[index]
        image = Image.open(image_path).convert("RGB")

        txt_path = image_path.with_suffix(".txt")
        caption = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else "a photo of mamaplugxs"
        caption = f"{caption}, {self.instance_prompt}"

        return {
            "input_ids": tokenize_prompt(self.tokenizer, caption),
            "pixel_values": self.image_transforms(image),
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--instance_data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="flux-lora")
    p.add_argument("--instance_prompt", type=str, default="mamaplugxs")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=500)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_warmup_steps", type=int, default=100)
    p.add_argument("--checkpointing_steps", type=int, default=100)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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

    device = accelerator.device
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

    # tokenizers
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # dataset
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
    )

    # 4-bit transformer
    from diffusers import FluxTransformer2DModel
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
    transformer = get_peft_model(transformer, LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        task_type=TaskType.FEATURE_EXTRACTION,
    ))
    transformer.print_trainable_parameters()

    # pipeline components (load once, then freeze & cast)
    pipe = FluxPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=weight_dtype)
    text_encoder = pipe.text_encoder.to(device, dtype=weight_dtype)
    text_encoder_2 = pipe.text_encoder_2.to(device, dtype=weight_dtype)
    vae = pipe.vae.to(device, dtype=weight_dtype)
    noise_scheduler = pipe.scheduler
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)

    # optimizer
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate)
    from diffusers.optimization import get_cosine_schedule_with_warmup
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
    )

    # accelerate prepare
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # encode prompt helper
    def encode_prompt(input_ids):
        t5_out = text_encoder(input_ids.to(device), return_dict=False)[0]
        clip_out = text_encoder_2(input_ids.to(device), return_dict=False)[0]
        return t5_out, clip_out[0]  # pooled

    # training loop
    global_step = 0
    for epoch in range(1):
        for step, batch in enumerate(train_dataloader):
            transformer.train()
            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)

            # 1. vae encode -> latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # 2. noise & timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 3. text embeds
            prompt_embeds, pooled_prompt_embeds = encode_prompt(batch["input_ids"])

            # 4. transformer noise prediction
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
                        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        transformer.save_pretrained(ckpt_dir)
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    # final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = accelerator.unwrap_model(transformer)
        transformer.save_pretrained(args.output_dir)
        from safetensors.torch import save_file
        save_file(transformer.state_dict(), os.path.join(args.output_dir, "pytorch_lora_weights.safetensors"))
        logger.info(f"LoRA saved to {args.output_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
