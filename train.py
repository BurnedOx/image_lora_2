#!/usr/bin/env python3
"""
lora_flux1_dev_mamaplugxs_tqdm.py
Fine-tune FLUX.1-dev with LoRA on 20 images on a Tesla T4.
Trigger word: mamaplugxs
Shows progress bars with tqdm
"""

import os, json, torch
from pathlib import Path
from datasets import Dataset
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import CLIPTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

# ------------------------------------------------------------------
# 1. Hyper-parameters
# ------------------------------------------------------------------
RESOLUTION      = 512
RANK            = 4
ALPHA           = 4
LR              = 1e-4
BATCH_SIZE      = 1
GRAD_ACC        = 4
MAX_STEPS       = 500
SAVE_STEPS      = 100
OUTPUT_DIR      = "flux_lora_mamaplugxs"
DATA_ROOT       = "data"
TRIGGER_WORD    = "mamaplugxs"

# ------------------------------------------------------------------
# 2. Dataset loader (adds trigger word)
# ------------------------------------------------------------------
def load_dataset(data_root: str):
    metadata_file = Path(data_root) / "metadata.jsonl"
    if not metadata_file.exists():
        raise FileNotFoundError("Create data/metadata.jsonl with {'file_name': '1.jpg', 'text': 'your caption'}")

    samples = []
    with open(metadata_file) as f:
        for line in f:
            samples.append(json.loads(line))

    def transform(ex):
        img_path = Path(data_root) / "img" / ex["file_name"]
        image = Image.open(img_path).convert("RGB")
        image = transforms.CenterCrop(RESOLUTION)(transforms.Resize(RESOLUTION)(image))
        ex["image"] = image
        ex["text"] = f"{TRIGGER_WORD} {ex['text']}"
        return ex

    ds = Dataset.from_list(samples)
    ds = ds.map(transform, remove_columns=["file_name"])
    ds = ds.with_format("torch")
    return ds

# ------------------------------------------------------------------
# 3. Build models
# ------------------------------------------------------------------
def build_models():
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
    pipe.enable_sequential_cpu_offload()
    transformer: FluxTransformer2DModel = pipe.transformer
    return pipe, transformer

# ------------------------------------------------------------------
# 4. Add LoRA
# ------------------------------------------------------------------
def add_lora(transformer):
    lora_conf = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2"
        ],
        lora_dropout=0.0,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_conf)
    transformer.print_trainable_parameters()
    return transformer

# ------------------------------------------------------------------
# 5. Training with tqdm
# ------------------------------------------------------------------
def main():
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACC)
    dataset = load_dataset(DATA_ROOT)
    pipe, transformer = build_models()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    transformer.enable_gradient_checkpointing()
    transformer = add_lora(transformer)
    transformer.train().to(accelerator.device)

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    global_step = 0

    progress_bar = tqdm(range(MAX_STEPS), disable=not accelerator.is_main_process)
    progress_bar.set_description("Steps")

    while global_step < MAX_STEPS:
        for batch in dataloader:
            images = batch["image"].to(accelerator.device, dtype=torch.float16)
            captions = batch["text"]

            text_inputs = tokenizer(
                captions, max_length=77, padding="max_length", truncation=True, return_tensors="pt"
            ).to(accelerator.device)
            prompt_embeds = pipe.text_encoder(text_inputs.input_ids)[0]

            bsz = images.shape[0]
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, 1000, (bsz,), device=images.device).long()

            with accelerator.accumulate(transformer):
                noisy = pipe.scheduler.add_noise(images, noise, timesteps)
                model_pred = transformer(noisy, timesteps, encoder_hidden_states=prompt_embeds).sample
                loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())
                if global_step % SAVE_STEPS == 0 or global_step == MAX_STEPS:
                    save_path = f"{OUTPUT_DIR}/checkpoint-{global_step}"
                    transformer.save_pretrained(save_path, safe_serialization=True)

            if global_step >= MAX_STEPS:
                break

    transformer.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    print("Done! LoRA saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()

