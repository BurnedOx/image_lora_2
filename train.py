import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers import DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import os
import json
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import Dataset
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

# Configuration
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
TRIGGER_WORD = "mamaplugxs"
OUTPUT_DIR = "./lora_finetuned_model"
BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
RESOLUTION = 512

# Initialize accelerator
accelerator = Accelerator()

# Load models and components
print("Loading models...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

# Freeze models except for LoRA parameters
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# Add LoRA layers to UNet using PEFT library
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
)
unet = get_peft_model(unet, lora_config)

# Prepare optimizer (only train LoRA parameters)
optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

# Load training data from data folder
def load_dataset_from_folder():
    # Read metadata
    metadata_path = os.path.join("data", "metadata.jsonl")
    image_dir = os.path.join("data", "img")
    
    images = []
    input_ids_list = []
    
    with open(metadata_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            file_name = data["file_name"]
            caption = data["text"]
            
            # Load and preprocess image
            image_path = os.path.join(image_dir, file_name)
            try:
                image = Image.open(image_path).convert("RGB")
                # Resize and convert to tensor
                image = image.resize((RESOLUTION, RESOLUTION))
                image = np.array(image) / 255.0  # Normalize to [0, 1]
                # Convert to tensor and change to channel-first format
                image = torch.from_numpy(image).permute(2, 0, 1).float()
                # Normalize to [-1, 1] range expected by VAE
                image = (image - 0.5) * 2.0
                images.append(image)
                
                # Tokenize caption and ensure it's a tensor
                tokenized = tokenizer(
                    caption,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = tokenized.input_ids[0]
                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids)
                input_ids_list.append(input_ids)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
    
    # Create a list of examples
    data = []
    for i in range(len(images)):
        data.append({
            "pixel_values": images[i],
            "input_ids": input_ids_list[i]
        })
    return Dataset.from_list(data)

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    input_ids = [item["input_ids"] for item in batch]
    
    # Stack tensors
    pixel_values = torch.stack(pixel_values)
    input_ids = torch.stack(input_ids)
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids
    }

train_dataset = load_dataset_from_folder()
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Prepare with accelerator
unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

# Training loop
print("Starting training...")
global_step = 0

for epoch in range(NUM_EPOCHS):
    unet.train()
    train_loss = 0.0
    
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
        # Move batch to device (tensors are already stacked by collate_fn)
        pixel_values = batch["pixel_values"].to(accelerator.device)
        input_ids = batch["input_ids"].to(accelerator.device)
        
        # Convert images to latent space
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215

        # Sample noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (BATCH_SIZE,), device=latents.device)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get text embeddings
        encoder_hidden_states = text_encoder(input_ids)[0]

        # Predict noise
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Calculate loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        accelerator.backward(loss)
        
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss += loss.item()
        global_step += 1

    print(f"Epoch {epoch} - Average Loss: {train_loss / len(train_dataloader):.4f}")

# Save LoRA weights using PEFT
print("Saving model...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save the entire PEFT model (includes LoRA weights and adapter config)
unet.save_pretrained(OUTPUT_DIR)

print(f"Training complete! LoRA weights saved to {OUTPUT_DIR}")
print(f"Use trigger word: '{TRIGGER_WORD}' when generating images")

# Test generation function
def generate_test_image(prompt):
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to("cuda")
    
    # Load LoRA weights (you'd need proper LoRA integration)
    # This is simplified - actual implementation would merge weights
    
    image = pipe(f"{TRIGGER_WORD} {prompt}").images[0]
    image.save("./test_output.png")
    return image

# Uncomment to test after training
# generate_test_image("a beautiful landscape")
