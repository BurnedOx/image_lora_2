import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import os

# Configuration
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "./lora_finetuned_model"
TRIGGER_WORD = "mamaplugxs"
OUTPUT_IMAGE_PATH = "./generated_image.png"

# Load base model
print("Loading base model...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    safety_checker=None,  # Disable safety checker for easier testing
).to("cuda")

# Load LoRA weights
print("Loading LoRA weights...")
pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)

# Generate image
prompt = f"{TRIGGER_WORD} a beautiful portrait"
print(f"Generating image with prompt: '{prompt}'")

with torch.no_grad():
    image = pipe(
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=512,
        width=512
    ).images[0]

# Save the image
image.save(OUTPUT_IMAGE_PATH)
print(f"Image saved to {OUTPUT_IMAGE_PATH}")