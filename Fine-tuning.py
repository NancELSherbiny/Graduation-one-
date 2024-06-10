import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
import gc
from torch.utils.checkpoint import checkpoint
from torch.optim import AdamW
from tqdm.auto import tqdm

# Define the model ID and dataset name
model_id = "CompVis/stable-diffusion-v1-4"
dataset_name = "Kareem99/data77"  # replace with your dataset name

# Load the tokenizer from the correct subfolder
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

# Load the dataset from Hugging Face
dataset = load_dataset(dataset_name)

# Define the image transformation
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Reduce image size to 256x256
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Prepare the dataset
def preprocess(example):
    prompt = example['caption']
    image = example['image']

    # Ensure the image is converted to RGB (some images might be grayscale or have alpha channels)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Apply the transformation
    image = image_transform(image)

    inputs = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = inputs.input_ids.squeeze(0)  # Remove the batch dimension
    # Return both input_ids and pixel_values as tensors
    return {"input_ids": input_ids, "pixel_values": image}

# Apply preprocessing
dataset = dataset.map(preprocess, remove_columns=["caption"])

# Function to collate the data properly
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]

    # Ensure all items are tensors
    input_ids = [torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in input_ids]
    pixel_values = [torch.tensor(item) if not isinstance(item, torch.Tensor) else item for item in pixel_values]

    # Convert lists to tensors
    input_ids = torch.stack(input_ids)
    pixel_values = torch.stack(pixel_values)

    return {"input_ids": input_ids, "pixel_values": pixel_values}

# Create DataLoader with a smaller batch size
train_dataloader = DataLoader(dataset['train'], batch_size=1, shuffle=True, collate_fn=collate_fn)

# Load model components
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Move components to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_encoder.to(device)
vae.to(device)
unet.to(device)

# Initialize the scaler
scaler = GradScaler()

# Set up optimizer
optimizer = AdamW(unet.parameters(), lr=5e-5)

# Gradient accumulation steps
accumulation_steps = 4

# Define a custom forward function to use checkpointing
def custom_forward(*inputs):
    return unet(*inputs).sample

# Training loop
num_epochs = 3
unet.train()

for epoch in range(num_epochs):
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    optimizer.zero_grad()
    for i, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast():
            # Forward pass
            latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(batch["input_ids"]).last_hidden_state

            # Predict the noise residual using checkpointing
            noise_pred = checkpoint(custom_forward, noisy_latents, timesteps, encoder_hidden_states)

            loss = torch.nn.functional.mse_loss(noise_pred, noise) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Clear variables and cache
            del latents, noise, noisy_latents, encoder_hidden_states, noise_pred
            torch.cuda.empty_cache()
            gc.collect()

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item() * accumulation_steps)  # Scale loss back to original value

print("Training complete!")

# Load the fine-tuned model into the pipeline
pipeline = StableDiffusionPipeline(
    vae=vae, text_encoder=text_encoder, unet=unet, scheduler=scheduler, tokenizer=tokenizer
).to(device)

prompt = "Your custom prompt"
generated_image = pipeline(prompt).images[0]
generated_image.save("generated_image.png")