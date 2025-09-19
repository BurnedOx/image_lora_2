import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers import DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import os
import json
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

# Configuration
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
TRIGGER_WORD = "mamaplugxs"
OUTPUT_DIR = "./lora_finetuned_model"
BATCH_SIZE = 2  # Increased from 1 for better gradient stability
NUM_EPOCHS = 100
LEARNING_RATE = 5e-6  # Further reduced learning rate for small batch
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
    r=8,  # Reduced rank to prevent overfitting
    lora_alpha=16,  # Adjusted alpha for better balance
    target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"],
    lora_dropout=0.1,
)
unet = get_peft_model(unet, lora_config)

# Prepare optimizer (only train LoRA parameters)
optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

# Custom PyTorch Dataset class
class ImageCaptionDataset(Dataset):
    def __init__(self, metadata_path, image_dir, tokenizer, resolution=512):
        self.metadata_path = metadata_path
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.resolution = resolution
        
        # Load metadata
        self.data = []
        with open(metadata_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        file_name = item["file_name"]
        caption = item["text"]
        
        try:
            # Load and preprocess image
            image_path = os.path.join(self.image_dir, file_name)
            image = Image.open(image_path).convert("RGB")
            # Resize and convert to tensor
            image = image.resize((self.resolution, self.resolution))
            image = np.array(image) / 255.0  # Normalize to [0, 1]
            # Convert to tensor and change to channel-first format
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            # Normalize to [-1, 1] range expected by VAE
            image = (image - 0.5) * 2.0
            
            # Tokenize caption
            tokenized = self.tokenizer(
                caption,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = tokenized.input_ids[0]
            
            return {
                "pixel_values": image,
                "input_ids": input_ids
            }
        except Exception as e:
            print(f"ERROR: Failed to load image {image_path}: {e}")
            # Return a dummy tensor to avoid breaking the batch
            dummy_image = torch.zeros((3, self.resolution, self.resolution))
            dummy_input_ids = torch.zeros((self.tokenizer.model_max_length,), dtype=torch.long)
            return {
                "pixel_values": dummy_image,
                "input_ids": dummy_input_ids
            }

# Load training data
train_dataset = ImageCaptionDataset(
    metadata_path=os.path.join("data", "metadata.jsonl"),
    image_dir=os.path.join("data", "img"),
    tokenizer=tokenizer,
    resolution=RESOLUTION
)

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

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Prepare with accelerator
unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
# Move VAE and text_encoder to accelerator device
vae.to(accelerator.device)
text_encoder.to(accelerator.device)

# Training loop with validation and early stopping
print("Starting training...")
global_step = 0
best_loss = float('inf')
patience = 5
patience_counter = 0

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
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss += loss.item()
        global_step += 1

    avg_loss = train_loss / len(train_dataloader)
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    
    # Early stopping check
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        # Save best model
        unet.save_pretrained(OUTPUT_DIR)
        print(f"New best model saved with loss: {best_loss:.4f}")
    else:
        patience_counter += 1
        print(f"Loss not improved. Patience: {patience_counter}/{patience}")
        
    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch} epochs")
        break

# Save LoRA weights using PEFT
print("Saving model...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save the entire PEFT model (includes LoRA weights and adapter config)
unet.save_pretrained(OUTPUT_DIR)

print(f"Training complete! LoRA weights saved to {OUTPUT_DIR}")
print(f"Use trigger word: '{TRIGGER_WORD}' when generating images")
