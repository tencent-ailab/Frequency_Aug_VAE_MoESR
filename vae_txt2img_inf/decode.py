import os
import glob
import tqdm
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch
from torch import autocast

from models.autoencoder import AutoencoderKLAFF


vae_ckpt = 'path/to/fa_vae.pth'
latent_path = "txt2img/latent_realv_512"
vae_name = 'AutoencoderKLAFF'
exp_name = 'vaeaff'
device_id = 0
res = 512


outpath = f"./txt2img/{vae_name}_{exp_name}_{res}"
if not os.path.exists(outpath):
    os.makedirs(outpath)


# define decoder
if exp_name == 'vaeaff':
    pretrain_config = "configs/autoencoder/vae_base_affunet_ffl.yaml"
else:
    NotImplementedError

config = OmegaConf.load(pretrain_config)
vae = AutoencoderKLAFF(**config.model.params)
device = torch.device('cuda', device_id)
vae = vae.to(device)

if vae_ckpt is not None:
    if vae_ckpt.endswith(".pth"):
        model_ckpt = torch.load(vae_ckpt)
    elif vae_ckpt.endswith(".ckpt"):
        model_ckpt = torch.load(vae_ckpt)["state_dict"]
else:
    model_ckpt = torch.load(os.path.join(vae_ckpt, 'model_last.pth'))

missing, unexpected = vae.load_state_dict(model_ckpt, strict=False)
print(f"Load {vae_ckpt} with {len(missing)} missing and {len(unexpected)} unexpected")
vae.eval()

# list latents
latent_files = glob.glob(os.path.join(latent_path, "*.pt"))

latents_batch = []
sublist_length = 6
for i in range(0, len(latent_files), sublist_length):
    batch = latent_files[i:i+sublist_length]
    latents_batch.append(batch)


num_samples = 1
img_index = len(os.listdir(outpath))
img_index = max(img_index, 0)

with torch.no_grad():
    with autocast("cuda"):
        for latent_names in tqdm.tqdm(latents_batch):
            # read latents
            latents = []
            for i in range(len(latent_names)):
                latent = torch.load(latent_names[i])[None,...]
                latents.append(latent)
            latents = torch.cat(latents, 0).to(device)
            
            # decode
            recons = vae.decode(latents)
            
            # save
            recon_imgs = np.clip(recons.detach().cpu().numpy(), -1, 1) 
            recon_imgs = ((recon_imgs + 1) * 0.5 * 255).astype(np.uint8)
            recon_imgs = np.transpose(recon_imgs, [0, 2, 3, 1])
            for i in range(recon_imgs.shape[0]):
                recon_img = Image.fromarray(recon_imgs[i])
                output_name = os.path.join(outpath, f'img_{img_index:04}.png')
                recon_img.save(output_name)
                img_index += 1
