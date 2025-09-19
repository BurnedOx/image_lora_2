import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
import os
from datetime import datetime

# Configuration
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "./lora_finetuned_model"
TRIGGER_WORD = "mamaplugxs"
OUTPUT_DIR = "./generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load base model
print("Loading base model...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

# Use better scheduler for improved quality
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Load LoRA weights
print("Loading LoRA weights...")
try:
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)
    print("LoRA weights loaded successfully")
except Exception as e:
    print(f"Error loading LoRA weights: {e}")
    exit(1)

# Generate multiple images with different prompts and parameters
prompts = [
    f"{TRIGGER_WORD} a beautiful portrait, high quality, detailed",
    f"{TRIGGER_WORD} smiling, professional photo, sharp focus",
    f"{TRIGGER_WORD} in elegant dress, cinematic lighting"
]

for i, prompt in enumerate(prompts):
    print(f"Generating image {i+1} with prompt: '{prompt}'")
    
    try:
        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=30,  # Reduced steps for faster generation
                guidance_scale=7.5,
                height=512,
                width=512,
                generator=torch.Generator(device="cuda").manual_seed(i)  # Different seed for variety
            ).images[0]

        # Save with timestamp and prompt info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}_{i+1}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        image.save(output_path)
        print(f"Image saved to {output_path}")
        
    except Exception as e:
        print(f"Error generating image {i+1}: {e}")
        continue

print("Generation completed!")