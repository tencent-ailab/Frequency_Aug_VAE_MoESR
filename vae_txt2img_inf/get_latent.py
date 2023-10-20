import os
import tqdm

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline


 
base_model_name = 'realv'
base_model = "path/to/your/base/model/Realistic_Vision_V5.1_noVAE"
device_id = 0
res = 512


outpath = f"./txt2img/latent_{base_model_name}_{res}"
if not os.path.exists(outpath):
    os.makedirs(outpath)

# define pipeline
pipe = StableDiffusionPipeline.from_pretrained(base_model, safety_checker=None).to("cuda")

# custom prompts
with open('./examples/prompts.txt', 'r') as file:
    lines = file.readlines()
prompts = [line.strip() for line in lines]
print('prompts:', len(prompts))

prompts_batch = []
sublist_length = 8
for i in range(0, len(prompts), sublist_length):
    batch = prompts[i:i+sublist_length]
    prompts_batch.append(batch)

num_samples = 1
img_index = len(os.listdir(outpath))
img_index = max(img_index, 0)

with torch.no_grad():
    with autocast("cuda"):
        for prompt in tqdm.tqdm(prompts_batch):
            generator = torch.Generator("cuda").manual_seed(42)

            feats = pipe(
                    prompt,
                    num_images_per_prompt=num_samples,
                    height=res,
                    width=res,
                    guidance_scale=7.5,
                    num_inference_steps=50,
                    generator=generator,
                    output_type='latent'
                ).images # (N, 4, 64, 64)
            feats = 1 / 0.1825 * feats # scale

            for i in range(feats.shape[0]):
                 # save latents
                latent_name = os.path.join(outpath, f'img_{img_index:04}.pt')
                torch.save(feats[i], latent_name)
                img_index += 1
