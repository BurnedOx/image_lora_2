#!/usr/bin/env python3
"""
Flux-1-Dev LoRA Fine-tuning Script
Optimized for RTX 4060 with 20GB RAM
Trigger word: mamaplugxs
"""

import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import Dataset
from diffusers import FluxTransformer2DModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

# ------------------------------------------------------------------
# 1. Hyper-parameters optimized for RTX 4060
# ------------------------------------------------------------------
RESOLUTION = 512
RANK = 8  # Higher rank for better quality on RTX 4060
ALPHA = 16
LR = 1e-4
BATCH_SIZE = 1
GRAD_ACC = 8  # Increased for better memory management
MAX_STEPS = 100  # More steps for better convergence
SAVE_STEPS = 10
WARMUP_STEPS = 10
OUTPUT_DIR = "flux_lora_mamaplugxs"
DATA_ROOT = "data"
TRIGGER_WORD = "mamaplugxs"

# ------------------------------------------------------------------
# 2. Dataset loader with trigger word
# ------------------------------------------------------------------
def load_dataset(data_root: str):
    metadata_file = Path(data_root) / "metadata.jsonl"
    if not metadata_file.exists():
        raise FileNotFoundError("Create data/metadata.jsonl with {'file_name': '1.jpg', 'text': 'your caption'}")

    samples = []
    with open(metadata_file) as f:
        for line in f:
            samples.append(json.loads(line))

    # Image transformations
    transform_img = transforms.Compose([
        transforms.Resize(RESOLUTION),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def transform(ex):
        img_path = Path(data_root) / "img" / ex["file_name"]
        image = Image.open(img_path).convert("RGB")
        image = transform_img(image)
        ex["pixel_values"] = image
        ex["input_ids"] = f"{TRIGGER_WORD} {ex['text']}"
        return ex

    ds = Dataset.from_list(samples)
    ds = ds.map(transform, remove_columns=["file_name"])
    ds = ds.with_format("torch")
    return ds

# ------------------------------------------------------------------
# 3. Build models with proper initialization
# ------------------------------------------------------------------
def build_models():
    # Load Flux transformer with memory optimization
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    
    # Load text encoder
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # Load scheduler
    scheduler = DDPMScheduler.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="scheduler"
    )
    
    return text_encoder, transformer, scheduler

# ------------------------------------------------------------------
# 4. Add LoRA with optimized configuration
# ------------------------------------------------------------------
def add_lora(transformer):
    lora_conf = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2"
        ],
        lora_dropout=0.1,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_conf)
    transformer.print_trainable_parameters()
    return transformer

# ------------------------------------------------------------------
# 5. Training loop with proper diffusion process
# ------------------------------------------------------------------
def main():
    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACC,
        mixed_precision="fp16"
    )
    
    # Load components
    dataset = load_dataset(DATA_ROOT)
    text_encoder, transformer, scheduler = build_models()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    # Enable gradient checkpointing and add LoRA
    transformer.enable_gradient_checkpointing()
    transformer = add_lora(transformer)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        transformer.parameters(), 
        lr=LR,
        weight_decay=1e-2
    )
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=MAX_STEPS - WARMUP_STEPS
    )
    
    # Data loader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        drop_last=True,
        num_workers=2
    )
    
    # Prepare models with accelerator
    text_encoder = text_encoder.to(accelerator.device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    global_step = 0
    
    # Progress bar
    progress_bar = tqdm(range(MAX_STEPS), disable=not accelerator.is_main_process)
    progress_bar.set_description("Training Progress")
    
    # Training loop
    transformer.train()
    for epoch in range(MAX_STEPS // len(dataloader) + 1):
        for batch in dataloader:
            if global_step >= MAX_STEPS:
                break
                
            # Get batch data
            images = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
            captions = batch["input_ids"]
            
            # Tokenize captions
            text_inputs = tokenizer(
                captions, 
                max_length=77, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            ).to(accelerator.device)
            
            # Get text embeddings
            with torch.no_grad():
                prompt_embeds = text_encoder(text_inputs.input_ids)[0]
            
            # Sample noise and timesteps
            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, 
                (images.shape[0],), 
                device=images.device
            ).long()
            
            # Add noise to images
            noisy_images = scheduler.add_noise(images, noise, timesteps)
            
            # Predict noise
            with accelerator.accumulate(transformer):
                model_pred = transformer(
                    hidden_states=noisy_images,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps
                ).sample
                
                # Calculate loss
                loss = F.mse_loss(model_pred, noise, reduction="mean")
                
                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            global_step += 1
            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Save checkpoint
                if global_step % SAVE_STEPS == 0:
                    save_path = f"{OUTPUT_DIR}/checkpoint-{global_step}"
                    transformer.save_pretrained(save_path, safe_serialization=True)
                    print(f"\nCheckpoint saved at step {global_step}")
    
    # Final save
    transformer.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    print(f"\nTraining complete! LoRA saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
