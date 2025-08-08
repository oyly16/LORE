import math
import random
import copy
from typing import Callable

import torch
import numpy as np
from einops import rearrange, repeat
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .model_lore import Flux
from .modules.conditioner_lore import HFEmbedder

def prepare_tokens(t5, source_prompt, target_prompt, replacements,show_tokens=False):
    _, _, src_dif_ids, tgt_dif_ids=t5.get_text_embeddings_with_diff(source_prompt,target_prompt,replacements,show_tokens=show_tokens)
    return src_dif_ids,tgt_dif_ids

transform = transforms.ToTensor()

def get_mask_one_tensor(mask_dirs,width,height,device):
    res = []
    for mask_dir in mask_dirs:
        mask_image = Image.open(mask_dir).convert('L')
        # resize
        mask_image = mask_image.resize((math.ceil(height/16), math.ceil(width/16)), Image.Resampling.LANCZOS)
        mask_tensor = transform(mask_image)
        mask_tensor = mask_tensor.squeeze(0)
        # to one dim
        mask_tensor = mask_tensor.flatten()
        mask_tensor = mask_tensor.to(device)
        res.append(mask_tensor)
    res = sum(res)
    res = res.view(1, 1, -1, 1)
    res = res.to(torch.bfloat16)
    return res

def get_v_mask(mask_dirs,width,height,device,txt_length=512):
    res = []
    for mask_dir in mask_dirs:
        mask_image = Image.open(mask_dir).convert('L')
        # resize
        mask_image = mask_image.resize((math.ceil(height/16), math.ceil(width/16)), Image.Resampling.LANCZOS)
        mask_tensor = transform(mask_image)
        mask_tensor = mask_tensor.squeeze(0)
        # to one dim
        mask_tensor = mask_tensor.flatten()
        mask_tensor = mask_tensor.to(device)
        res.append(mask_tensor)
    res = sum(res)
    res = torch.cat([torch.ones(txt_length).to(device),res])
    res = res.view(1, 1, -1, 1)
    res = res.to(torch.bfloat16)
    return res

def add_masked_noise_to_z(z,mask,width,height,seed=42,noise_scale=0.1):
    if noise_scale == 0:
        return z
    noise = torch.randn(z.shape,device=z.device,dtype=z.dtype,generator=torch.Generator(device=z.device).manual_seed(seed))
    if noise_scale > 10:
        return noise
    # how to change z?
    z = z*(1-mask[0])+noise_scale*noise*mask[0]+(1-noise_scale)*z*mask[0]
    return z

def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0,
    trainable_noise_list=None,
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])
    

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    
    step_list = []
    attn_map_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]
        # when editing add optim latent for several steps
        if trainable_noise_list and i != 0 and i<len(trainable_noise_list):
            # smask = info['source_mask'].squeeze(0)
            # img = trainable_noise_list[i]*smask+img*(1-smask)
            img = trainable_noise_list[i]

        pred, info, attn_maps_mid = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info, attn_maps = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )

        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

        # return attnmaps L,1,512,N
        step_list.append(t_curr)
        attn_map_list.append((attn_maps_mid+attn_maps)/2)

    attn_map_list = torch.stack(attn_map_list)
    return img, info, step_list, attn_map_list

selected_layers = range(8,44)

def gaussian_smooth(attnmap,wh,kernel_size=3,sigma=0.5):
    # to 2d
    attnmap = rearrange(
                        attnmap,
                        "b (w h) -> b (w) (h)",
                        w=math.ceil(wh[0]/16),
                        h=math.ceil(wh[1]/16),
                    )
    attnmap = attnmap.unsqueeze(1)
    # prepare kernel
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=attnmap.device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.to(dtype=attnmap.dtype)
    # gaussian smooth
    attnmap_smoothed = F.conv2d(attnmap, kernel, padding=kernel_size // 2)
    return attnmap_smoothed.view(attnmap_smoothed.shape[0], -1)

def compute_attn_max_loss(attnmaps,source_mask,wh):
    # attnmaps L,1,N,k
    attnmaps = attnmaps[selected_layers,0,:,:]
    attnmaps = attnmaps.mean(dim=-1)
    src_mask = source_mask.view(-1).unsqueeze(0)
    p = attnmaps*src_mask
    p = gaussian_smooth(p, wh, kernel_size=3, sigma=0.5)
    p = p.max(dim=1).values
    loss = (1 - p).mean()
    return loss

def compute_attn_min_loss(attnmaps,source_mask,wh):
    # attnmaps L,1,N,k
    attnmaps = attnmaps[selected_layers,0,:,:]
    attnmaps = attnmaps.mean(dim=-1)
    src_mask = source_mask.view(-1).unsqueeze(0)
    p = attnmaps*src_mask
    p = gaussian_smooth(p, wh, kernel_size=3, sigma=0.5)
    p = p.max(dim=1).values 
    loss = p.mean()
    return loss

def denoise_with_noise_optim(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # loss cal
    token_ids: list[list[int]],
    source_mask: Tensor,
    training_steps: int,
    training_epochs: int,
    learning_rate: float,
    seed: int,
    noise_scale: float,
    # sampling parameters
    timesteps: list[float],
    info, 
    guidance: float = 4.0
):
    # this is ignored for schnell
    #print(f'training the noise in last {training_steps} steps and {training_epochs} epochs')
    #timesteps = timesteps[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    attn_map_list = []
    trainable_noise_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        if i >= training_steps:
            break
        # prepare ori parameters
        ori_txt = txt.clone()
        ori_img = img.clone()
        ori_vec = vec.clone()
        
        # prepare trainable noise
        if i == 0:
            if noise_scale == 0:
                trainable_noise = torch.nn.Parameter(img.clone().detach(), requires_grad=True)
            else:
                noise = torch.randn(img.shape,device=img.device,dtype=img.dtype,generator=torch.Generator(device=img.device).manual_seed(seed))
                noise = img*(1-source_mask[0])+ noise_scale*noise*source_mask[0] + (1-noise_scale)*img*source_mask[0]
                trainable_noise = torch.nn.Parameter(noise.clone().detach(), requires_grad=True)
        else:
            trainable_noise = torch.nn.Parameter(img.clone().detach(), requires_grad=True)
        optimizer = optim.Adam([trainable_noise], lr=learning_rate)

        # run one training step
        for j in range(training_epochs):
            optimizer.zero_grad()
            txt = ori_txt.clone().detach()
            vec = ori_vec.clone().detach()
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            info['t'] = t_prev
            info['inverse'] = False
            info['second_order'] = False
            info['inject'] = False # tried True, seems not necessary
            pred, info, attn_maps_mid = model(
                img=trainable_noise,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                info=info
            )
            
    
            img_mid = trainable_noise + (t_prev - t_curr) / 2 * pred
    
            t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
            info['second_order'] = True
            pred_mid, info, attn_maps = model(
                img=img_mid,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_mid,
                guidance=guidance_vec,
                info=info
            )
    
            first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
            img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order
    
            # attnmaps L,1,N,512 for cal loss
            attn_maps=(attn_maps_mid+attn_maps)/2
            total_loss = 0.0
            for indices,change,ratio in token_ids:
                if change:
                    total_loss += compute_attn_max_loss(attn_maps[:,:,:,indices], source_mask, info['wh'])
                else:
                    if ratio != 0:
                        total_loss += ratio*compute_attn_min_loss(attn_maps[:,:,:,indices], source_mask, info['wh'])
            total_loss.backward()
            with torch.no_grad():
                trainable_noise.grad *= source_mask[0]
            optimizer.step()
            print(f"Time {t_curr:.4f} Step {j+1}/{training_epochs}, Loss: {total_loss.item():.6f}")
            
        attn_map_list.append(attn_maps.detach())
        step_list.append(t_curr)
        trainable_noise = trainable_noise.detach()
        trainable_noise_list.append(trainable_noise.clone())

    attn_map_list = torch.stack(attn_map_list)
    return img, info, step_list, attn_map_list, trainable_noise_list

def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
