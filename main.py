import os
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
# from diffusers import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
)
from diffusers import DDPMScheduler, AutoencoderKL
import torch
from torchvision import transforms
import logging

# Suppress logs for specific libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
# Set base paths
base_path = 'yisol/IDM-VTON'

# Load models and components
unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
    ignore_mismatched_sizes=True,
).eval()

unet.requires_grad_(False)

tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer", use_fast=False)
tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2", use_fast=False)

noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=torch.float16).eval()
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path, subfolder="text_encoder_2", torch_dtype=torch.float16
).eval()

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path, subfolder="image_encoder", torch_dtype=torch.float16
).eval()

vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=torch.float16).eval()

UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path, subfolder="unet_encoder", torch_dtype=torch.float16
).eval()

# Ensure models do not require gradients
for model in [unet, vae, text_encoder_one, text_encoder_two, image_encoder, UNet_Encoder]:
    model.requires_grad_(False)

# Load the pipeline
pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)

pipe.unet_encoder = UNet_Encoder

# Transform for tensor conversion
tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Function to perform try-on and save the output
def try_on(
    human_image_path, garment_image_path, garment_description, output_path, pose_image_path, mask_image_path,
    denoise_steps=30, seed=42
):
    device = "cuda"

    # Define cross_attention_kwargs for scale
    cross_attention_kwargs = {"scale": 1.0}

    # Move models to device
    pipe.to(device)
    pipe.unet_encoder.to(device)

    # Load and process images
    human_img = Image.open(human_image_path).convert("RGB").resize((768, 1024))
    garm_img = Image.open(garment_image_path).convert("RGB").resize((768, 1024))
    pose_img = Image.open(pose_image_path).convert("RGB").resize((768, 1024))
    mask = Image.open(mask_image_path).convert("RGB").resize((768, 1024))

    # Generate try-on image
    with torch.no_grad():
        prompt = "model is wearing " + garment_description
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
            prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt
        )

        garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16)
        generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

        images = pipe(
            prompt_embeds=prompt_embeds.to(device, torch.float16),
            negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
            pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
            num_inference_steps=denoise_steps,
            generator=generator,
            strength=1.0,
            pose_img=tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16),
            text_embeds_cloth=prompt_embeds.to(device, torch.float16),
            cloth=garm_tensor,
            mask_image=mask,
            image=human_img,
            height=1024,
            width=768,
            ip_adapter_image=garm_img.resize((768, 1024)),
            guidance_scale=2.0,
            cross_attention_kwargs=cross_attention_kwargs,
        )[0]

    # Save the generated image
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "swapped_garment.png")
    images[0].save(output_file)
    print(f"Swapped garment image saved to {output_file}")

# Example usage
try_on(
    human_image_path="Images/Human_Image.jpg",
    garment_image_path="Images/Garment_Image.jpg",
    garment_description="Short Sleeve V Neck deep Blue T-shirt",
    output_path="Images",
    pose_image_path="Images/Pose_Image.png",
    mask_image_path="Images/Mask_Image.png"
)
