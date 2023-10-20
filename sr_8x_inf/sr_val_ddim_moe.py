import os
import argparse
import math
import copy
import cv2
from PIL import Image
from tqdm import trange
import numpy as np
from itertools import islice
from omegaconf import OmegaConf
from einops import rearrange
import torch
from torch import autocast
from torch.distributed import init_process_group
from pytorch_lightning import seed_everything
from contextlib import nullcontext

from basicsr.utils.imresize import imresize
from basicsr.utils import img2tensor
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_moe import DDIMMoeSampler
from ldm.models.vqmodel import VQModelResiUnet


def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_models_from_config(config, ckpts, main_device, verbose=False):
    init_model = instantiate_from_config(config.model)
    models = [copy.deepcopy(init_model) for i in range(4)]
    
    for i in range(len(models)):
        model = models[i]
        
        if len(ckpts) > 0:
            ckpt = ckpts[i]
            print(f"Loading model from {ckpt}")
            sd = torch.load(ckpt, map_location="cpu")

            m, u = model.load_state_dict(sd, strict=False)
            if len(m) > 0 and verbose:
                print("missing keys:")
                print(m)
            if len(u) > 0 and verbose:
                print("unexpected keys:")
                print(u)

        model.to(main_device)
        model.eval()

        model.model_ema.store(model.model.parameters())
        model.model_ema.copy_to(model.model)

    return models
    

def load_img(path, factor, return_upscale=False):
    '''
    Input:
        return_upscale: we need to upscale images first to get encoding latents
    Return: 
        image: Tensor, [-1, 1], conditions that will concat with the latent for SR
    '''
    image = cv2.imread(path) # (bgr)
    h, w, _ = image.shape
    if return_upscale:
        vae_enc_input = imresize(image, output_shape=(h*factor, w*factor))
        vae_enc_input = img2tensor(vae_enc_input, bgr2rgb=True, float32=True) / 255.0
        vae_enc_input = 2.*vae_enc_input - 1.
        vae_enc_input = vae_enc_input[None, ...]

    if factor == 8:
        image = imresize(image, output_shape=(h*2, w*2))
    image = img2tensor(image, bgr2rgb=True, float32=True) / 255.0
    image = 2.*image - 1.
    image = image[None, ...]

    if return_upscale: 
        return image, vae_enc_input
    else:   
        return image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        help="path to the input image",
        default="/mnt/lustre/jywang/dataset/ImageSR/RealSRSet/"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/sr-samples"
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=8,
        help="downsampling factor, 4 or 8",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs='+',
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--save_input",
        action='store_true',
        help="if enabled, save inputs",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="input size",
    )
    parser.add_argument(
        "--color_fix",
        action='store_true',
        help="if enabled, use adain for color fix",
    )
    parser.add_argument(
		"--num_length",
		type=int,
		default=-1,
		help="length of each data split",
	)
    parser.add_argument(
		"--data_idx",
		type=int,
		help="index of data split",
	)
    parser.add_argument(
		"--device",
		type=int,
		help="index of data split",
	)
    parser.add_argument(
        "--decoder_config",
        type=str,
        default="",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--decoder_ckpt",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--return_latents",
        action='store_true',
        help="get diffusion latents before decoding",
    )
    parser.add_argument(
        "--skip_return_samples",
        action='store_true',
        help="whether get samples",
    )
    opt = parser.parse_args()
    return opt


def main():
    opt = get_args()
    init_process_group(backend='nccl')

    seed_everything(opt.seed)

    # load moe models 
    config = OmegaConf.load(f"{opt.config}")
    
    device = f"cuda:{opt.device}"
    models = []
    print("Models should be listed in an ascending order, ie. 0-250, 250-500...")
    models = load_models_from_config(config, opt.ckpt, device, verbose=True)

    # load vae_aff
    if opt.decoder_config:
        decoder_config = OmegaConf.load(f"{opt.decoder_config}")
        vae = VQModelResiUnet(ddconfig=decoder_config.model.params.ddconfig, n_embed=8192, embed_dim=3)
        vae_ckpt = torch.load(os.path.join(opt.decoder_ckpt))
        missing, unexpected = vae.load_state_dict(vae_ckpt, strict=False)
        print(f"Load {opt.decoder_ckpt} with {len(missing)} missing and {len(unexpected)} unexpected")
        print("unexpected:", unexpected)
        models[0].first_stage_model = vae
    
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    if opt.save_input:
        input_path = os.path.join(outpath, "inputs")
        os.makedirs(input_path, exist_ok=True)
    if opt.return_latents:
        npy_path = os.path.join(outpath, "diffusion_latents")
        os.makedirs(npy_path, exist_ok=True)

    batch_size = opt.n_samples
    
    # get evaluate subset
    num_length = opt.num_length
    try:
        img_list_old = sorted(os.listdir(opt.init_img),  key=lambda x:int(x.split(".")[0]))
    except:
        img_list_old = sorted(os.listdir(opt.init_img))
    if num_length > 0:
        print("Dataset Start Index:", opt.data_idx * num_length)
        img_list_old = img_list_old[opt.data_idx * num_length:min((1+opt.data_idx) * num_length, len(img_list_old))]
    img_list = sorted(img_list_old)
    niters = math.ceil(len(img_list) / batch_size)
    img_list_chunk = [img_list[i * batch_size:(i + 1) * batch_size] for i in range((len(img_list) + batch_size - 1) // batch_size )]

    sampler = DDIMMoeSampler(models, opt.device)

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    x_T = None

    # inference
    with torch.no_grad():
        with precision_scope("cuda"):
            for n in trange(niters, desc="Sampling"):
                cur_img_list = img_list_chunk[n]
                init_image_list = []
                vae_enc_input_list = []
                for item in cur_img_list:
                    cur_image, vae_enc_input  = load_img(os.path.join(opt.init_img, item), opt.factor, return_upscale=True)
                    cur_image = cur_image.to(device)
                    init_image_list.append(cur_image)

                    # prepare latent for vae decode
                    vae_enc_input = vae_enc_input.to(device)
                    vae_enc_input_list.append(vae_enc_input)

                init_image = torch.cat(init_image_list, dim=0)

                seed_everything(opt.seed)

                latents, _ = sampler.sample(t_enc, init_image.size(0), (3,opt.input_size // 4,opt.input_size // 4), init_image, eta=opt.ddim_eta, verbose=False, x_T=x_T)

                # decode latent
                if not opt.skip_return_samples:
                    models[0].first_stage_model.to(device)
                    latents = latents.to(device) # tensor
                    if isinstance(models[0].first_stage_model, VQModelResiUnet):
                        # upscale image and get encode feats
                        vae_enc_inputs = torch.cat(vae_enc_input_list, dim=0).to(device)
                        h, enc_fea = models[0].first_stage_model.encode_to_prequant(vae_enc_inputs)
                        x_samples,_,_ = models[0].first_stage_model.decode(latents, enc_fea)
                    else:
                        NotImplementedError

                    models[0].first_stage_model.to(device)     

                    if opt.color_fix:
                        x_samples = adaptive_instance_normalization(x_samples, init_image.to(x_samples.device))
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)   

                if opt.return_latents:
                    latents = latents.cpu().numpy()

                if not opt.skip_save:
                    for i in range(latents.shape[0]):
                        # save sr image
                        if not opt.skip_return_samples:
                            x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(sample_path, cur_img_list[i]))

                        # save latent
                        if opt.return_latents:
                            latent_name = cur_img_list[i].split(".")[0] + f".npy"
                            np.save(os.path.join(npy_path, latent_name), latents[i])
                    
                        # save input image
                        if opt.save_input:
                            x_input = 255. * rearrange(init_image[i].cpu().numpy(), 'c h w -> h w c')
                            x_input = (x_input+255.)/2
                            Image.fromarray(x_input.astype(np.uint8)).save(
                                os.path.join(input_path, cur_img_list[i]))

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

                        
if __name__ == "__main__":
    main()


