import torch
import base64
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

while True:
    prompt = input(">>> ")
    batch_size = 2 # Generate two batches of images
    images_batch = pipe(prompt=[prompt] * batch_size, num_inference_steps=1, guidance_scale=0.0).images
    for i, image in enumerate(images_batch):
        image.save(f"output{i+1}.png")
