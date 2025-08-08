import os
import re
import time
import math
from dataclasses import dataclass
from glob import iglob
import argparse
import torch
from einops import rearrange
from fire import Fire
from PIL import ExifTags, Image

from flux.sampling_lore import denoise, get_schedule, prepare, unpack, get_v_mask, add_masked_noise_to_z,get_mask_one_tensor, denoise_with_noise_optim,prepare_tokens
from flux.util_lore import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5)
from transformers import pipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import os
import json

def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image  

def get_token_ratio(tgt_token_ids,change_id):
    all_len = 0
    tgt_len = len(tgt_token_ids[change_id][1])
    for _,ids in tgt_token_ids:
        all_len += len(ids)
    return float(all_len)/float(tgt_len)

def run_one_sample_rf_inverse(src_img, src_prompt, steps, inject_step, t5, clip, model, ae):
    print(f"Inversing {src_img}, inje/step {inject_step}/{steps}")
    print(f"src prompt: {src_prompt}")
    torch_device = torch.device("cuda")
    init_image = np.array(Image.open(src_img).convert('RGB'))
    
    shape = init_image.shape

    new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
    new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16

    init_image = init_image[:new_h, :new_w, :]

    width, height = init_image.shape[0], init_image.shape[1]
    init_image = encode(init_image, torch_device, ae)
    t0 = time.perf_counter()

    info = {}
    info['feature_path'] = args.feature_path
    info['feature'] = {}
    info['inject_step'] = inject_step
    info['wh'] = (width, height)
    if not os.path.exists(args.feature_path):
        os.mkdir(args.feature_path)

    inp = prepare(t5, clip, init_image, prompt=src_prompt)
    timesteps = get_schedule(steps, inp["img"].shape[1], shift=True)
    info['x_ori'] = inp["img"].clone()

    # inversion initial noise
    torch.set_grad_enabled(False)
    z, info, step_list, attn_map_list = denoise(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)

    return init_image,z,info

def resize_image_and_mask(args):
    def resize_by_longside(image, target_long_side):
        w, h = image.size
        long_side = max(w, h)
        if long_side <= target_long_side:
            return None
        if h >= w:
            new_h = target_long_side
            new_w = int(w * target_long_side / h)
        else:
            new_w = target_long_side
            new_h = int(h * target_long_side / w)
        return image.resize((new_w, new_h), Image.LANCZOS)

    image = Image.open(args.source_img_dir).convert("RGB")
    resized_img = resize_by_longside(image, args.resize)
    if resized_img:
        base, _ = os.path.splitext(args.source_img_dir)
        new_img_path = f"{base}_resized.png"
        resized_img.save(new_img_path)
        args.source_img_dir = new_img_path
        print(f"Image resized and saved to {new_img_path}")
    else:
        print("Image does not need resizing.")
        return

    mask = Image.open(args.source_mask_dir).convert("L")
    resized_mask = resize_by_longside(mask, args.resize)
    if resized_mask:
        base, _ = os.path.splitext(args.source_mask_dir)
        new_mask_path = f"{base}_resized.png"
        resized_mask.save(new_mask_path)
        args.source_mask_dir = new_mask_path

def main(
    args,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    offload: bool = False,
    add_sampling_metadata: bool = True,
):
    name = args.name
    offload = args.offload
    if args.resize != -1:
        resize_image_and_mask(args)

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 25

    # init all components
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)
    for param in model.parameters():
        param.requires_grad = False  # freeze the model
    for param in t5.parameters():
        param.requires_grad = False  # freeze the model
    for param in clip.parameters():
        param.requires_grad = False  # freeze the model
    for param in ae.parameters():
        param.requires_grad = False  # freeze the model

    # run one sample

    # prepare save path
    os.makedirs(args.savedir,exist_ok=True)

    # run inverse
    t0 = time.perf_counter()  
    init_image,z_init,info=run_one_sample_rf_inverse(args.source_img_dir, args.source_prompt, args.num_steps, args.inject, t5, clip, model, ae)
    width,height = info['wh']

    # prepare arguments
    inp_optim = prepare(t5, clip, init_image, prompt=args.target_prompt)
    inp_target = prepare(t5, clip, init_image, prompt=args.target_prompt)
    v_mask = get_v_mask([args.source_mask_dir],width,height,device=torch_device)
    source_mask = get_mask_one_tensor([args.source_mask_dir],width,height,device=torch_device) 
    timesteps = get_schedule(args.num_steps, inp_optim["img"].shape[1], shift=True)
    info['change_v'] = args.v_inject # v_mask
    info['v_mask'] = v_mask
    info['source_mask'] = source_mask
    print(f'using source mask {args.source_mask_dir}')

    print(f"tgt prompt: {args.target_prompt}")
    # prepare token_ids
    token_ids=[]
    replacements = [[None,args.target_object,-1,int(args.target_index)]]
    src_dif_ids,tgt_dif_ids = prepare_tokens(t5, args.source_prompt, args.target_prompt, replacements,True)
    for t_ids in tgt_dif_ids:
        token_ids.append([t_ids,True,1])
    print('token_ids',token_ids)

    # run
    for seed in args.seeds:
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        print(f'optimizing & editing noise with seed {seed}, noise_scale {args.noise_scale}, training_steps {args.training_steps}, training_epochs {args.training_epochs}, learning_rate {args.learning_rate}')
        if args.training_epochs != 0:
            # run noise optim
            torch.set_grad_enabled(True)
            inp_optim["img"] = z_init
            _, info, step_list, attn_map_list, trainable_noise_list = denoise_with_noise_optim(model,**inp_optim,token_ids=token_ids,source_mask=source_mask,training_steps=args.training_steps,training_epochs=args.training_epochs,learning_rate=args.learning_rate,seed=seed,noise_scale=args.noise_scale,timesteps=timesteps,info=info,guidance=args.guidance)
            z_optim = trainable_noise_list[0]
        else:
            z_optim = add_masked_noise_to_z(z_init,source_mask,width,height,seed=seed,noise_scale=args.noise_scale)
            trainable_noise_list = None
        # run editing
        inp_target["img"] = z_optim
        timesteps = get_schedule(args.num_steps, inp_target["img"].shape[1], shift=True)
        model.eval()
        torch.set_grad_enabled(False)
        x, _, step_list, attn_map_list = denoise(model, **inp_target, timesteps=timesteps, guidance=args.guidance, inverse=False, info=info, trainable_noise_list = trainable_noise_list)  
            
        # decode latents to pixel space
        batch_x = unpack(x.float(), width,height)
    
        for x in batch_x:
            x = x.unsqueeze(0)
            
    
            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                x = ae.decode(x)
    
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # bring into PIL format and save
            x = x.clamp(-1, 1)
            x = embed_watermark(x.float())
            x = rearrange(x[0], "c h w -> h w c")
    
            img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
            exif_data = Image.Exif()
            exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
            exif_data[ExifTags.Base.Make] = "Black Forest Labs"
            output_path = os.path.join(args.savedir,f'{args.savename}_inj_{args.inject:02d}_{args.num_steps}_seed_{seed}_epoch_{args.training_epochs:03d}_scale_{args.noise_scale:.2f}.png')
            img.save(output_path, exif=exif_data, quality=95, subsampling=0)
    
    t1 = time.perf_counter()  
    print(f"Done in {t1 - t0:.1f}s. Saving {output_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='flux-dev', type=str,
                        help='flux model')
    parser.add_argument('--source_img_dir', default='', type=str,
                        help='The path of the source image')
    parser.add_argument('--source_mask_dir', default='', type=str,
                        help='The path of the source object mask')
    parser.add_argument('--source_prompt', type=str,
                        help='describe the content of the source image (or leaves it as null)')
    parser.add_argument('--target_prompt', type=str,
                        help='describe the requirement of editing')
    parser.add_argument('--target_object', type=str,
                        help='describe the target object')
    parser.add_argument('--target_index', type=int,
                        help='start index of target object')
    parser.add_argument('--feature_path', type=str, default='feature',
                        help='the path to save the feature ')
    parser.add_argument('--guidance', type=float, default=2,
                        help='guidance scale')
    parser.add_argument('--num_steps', type=int, default=15,
                        help='the number of timesteps for inversion and denoising')
    parser.add_argument('--inject', type=int, default=12,
                        help='the number of timesteps which apply the feature sharing')
    parser.add_argument('--seeds', default=[0,50], type=int,nargs='+',
                        help='seed')
    parser.add_argument('--noise_scale', default=0.9, type=float,
                        help='noise scale')
    parser.add_argument('--training_steps', default=1, type=int,
                        help='training_steps')
    parser.add_argument('--learning_rate', default=0.01, type=float,
                        help='training learning rate')
    parser.add_argument('--training_epochs', default=10, type=int,
                        help='training_epochs')
    parser.add_argument('--v_inject', default=2, type=int,
                        help='how to inject v. -1 for not inject. 0 for rf edit. 1 for debuged rf edit. 2 for masked inject (ours).')
    parser.add_argument('--savedir', default="outputs_demo", type=str,
                        help='save dir')
    parser.add_argument('--savename', default="demo", type=str,
                        help='save file name')
    parser.add_argument('--resize', default=800, type=int,
                        help='resize longside')
    parser.add_argument('--offload', action='store_true', help='set it to True if the memory of GPU is not enough')

    args = parser.parse_args()

    main(args)
