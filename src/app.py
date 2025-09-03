import os
import re
import time
import math
from dataclasses import dataclass
from glob import iglob
import argparse
from einops import rearrange
from PIL import ExifTags, Image
import torch
import gradio as gr
import numpy as np
import spaces
# from huggingface_hub import login
# login(token=os.getenv('Token'))
from flux.sampling_lore import denoise, get_schedule, prepare, unpack, get_v_mask, add_masked_noise_to_z,get_mask_one_tensor, denoise_with_noise_optim,prepare_tokens
from flux.util_lore import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5)

def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    ae.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image
from torchvision import transforms
transform = transforms.ToTensor()


model_name = 'flux-dev'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
offload = False

name = model_name
is_schnell = model_name == "flux-schnell"
save = True
output_dir = 'outputs_gradio'

t5 = load_t5(device, max_length=256 if name == "flux-schnell" else 512)
clip = load_clip(device)
model = load_flow_model(model_name, device=device)
ae = load_ae(name, device=device)
t5.eval()
clip.eval()
ae.eval()
info = {}

if offload:
    model.cpu()
    torch.cuda.empty_cache()
    ae.encoder.to(device)
for param in model.parameters():
    param.requires_grad = False  # freeze the model
for param in t5.parameters():
    param.requires_grad = False  # freeze the model
for param in clip.parameters():
    param.requires_grad = False  # freeze the model
for param in ae.parameters():
    param.requires_grad = False  # freeze the model

def resize_image(image, resize_longside):
    pil_image = Image.fromarray(image)
    h, w = pil_image.size[1], pil_image.size[0]
    if h <= resize_longside and w <= resize_longside:
        return image

    if h >= w:
        new_h = resize_longside 
        new_w = int(w * resize_longside  / h)
    else:
        new_w = resize_longside 
        new_h = int(h * resize_longside  / w)

    resized_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
    return np.array(resized_image)

def resize_mask(mask,height,width, resize_longside):
    pil_mask = Image.fromarray(mask.astype(np.uint8))  # ensure it's 8-bit for PIL
    resized_pil = pil_mask.resize((width, height), Image.NEAREST)  # width first!
    return np.array(resized_pil)

def inverse(brush_canvas,src_prompt, 
            inversion_num_steps, injection_num_steps, 
            inversion_guidance, resize_longside,
         ):
    print(f"Inversing {src_prompt}, guidance {inversion_guidance}, inje/step {injection_num_steps}/{inversion_num_steps}")
    # if info:
    #     del info        
    info = {'src_p':src_prompt}
    
    rgba_init_image = brush_canvas["background"]
    init_image = rgba_init_image[:,:,:3]
    if resize_longside > 0:
        init_image = resize_image(init_image,resize_longside)
    shape = init_image.shape

    new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
    new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16

    init_image = init_image[:new_h, :new_w, :]
    if save:
        ori_output_path = os.path.join(output_dir,f'{src_prompt[:20]}_ori.png')
        Image.fromarray(init_image,'RGB').save(ori_output_path)
        
    width, height = init_image.shape[0], init_image.shape[1]
    init_image = encode(init_image, device, ae)

    t0 = time.perf_counter()

    info['feature'] = {}
    info['inject_step'] = injection_num_steps
    info['wh'] = (width, height)

    inp = prepare(t5, clip, init_image, prompt=src_prompt)
    timesteps = get_schedule(inversion_num_steps, inp["img"].shape[1], shift=True)
    info['x_ori'] = inp["img"].clone()

    # inversion initial noise
    torch.set_grad_enabled(False)
    model.to(device)
    z0, info, _, _ = denoise(model, **inp, timesteps=timesteps, guidance=inversion_guidance, inverse=True, info=info)
    info = info
        
    t1 = time.perf_counter()
    print(f"inversion Done in {t1 - t0:.1f}s.")
    return z0,info

@spaces.GPU(duration=60)
def edit(brush_canvas, source_prompt, inversion_guidance,
            target_prompt, target_object,target_object_index,
            inversion_num_steps, injection_num_steps, 
            training_epochs, 
            denoise_guidance,noise_scale,seed,resize_longside
         ):
    resize_longside = int(resize_longside)
    torch.cuda.empty_cache()
    z0,info=inverse(brush_canvas,source_prompt, 
        inversion_num_steps, injection_num_steps, 
        inversion_guidance, resize_longside)
    
    rgba_init_image = brush_canvas["background"]
    rgba_mask = brush_canvas["layers"][0]
    init_image = rgba_init_image[:,:,:3]
    if resize_longside > 0:
        init_image = resize_image(init_image, resize_longside)
    width, height = info['wh']
    init_image = init_image[:width, :height, :]
    #rgba_init_image = rgba_init_image[:height, :width, :]
    
    if resize_longside > 0:
        mask = resize_mask(rgba_mask[:,:,3],height,width,resize_longside)
    else:            
        mask = rgba_mask[:width, :height, 3]
    mask = mask.astype(int)
    
    rgba_mask[:,:,3] = rgba_mask[:,:,3]//2
    masked_image = Image.alpha_composite(Image.fromarray(rgba_init_image, 'RGBA'), Image.fromarray(rgba_mask, 'RGBA'))
    masked_image = masked_image.resize((height, width), Image.LANCZOS)
        

    # prepare source mask and vmask
    init_image = encode(init_image, device, ae)
    inp_optim = prepare(t5, clip, init_image, prompt=target_prompt)
    inp_target = prepare(t5, clip, init_image, prompt=target_prompt)
    v_mask,source_mask = get_v_src_masks(mask,width,height,device)
    info['change_v'] = 2 # v_mask
    info['v_mask'] = v_mask
    info['source_mask'] = source_mask
    info['inject_step'] = injection_num_steps
    timesteps = get_schedule(inversion_num_steps, inp_optim["img"].shape[1], shift=True)
    seed = int(seed)
    if seed == -1:
        seed = torch.randint(0, 2**32, (1,)).item()

    # prepare token_ids
    token_ids=[]
    replacements = [[None,target_object,-1,int(target_object_index)]]
    src_dif_ids,tgt_dif_ids = prepare_tokens(t5, source_prompt, target_prompt, replacements,True)
    for t_ids in tgt_dif_ids:
        token_ids.append([t_ids,True,1])
    print('token_ids',token_ids)
    
    # do latent optim
    t0 = time.perf_counter() 
    print(f'optimizing & editing noise, {target_prompt} with seed {seed}, noise_scale {noise_scale}, training_epochs {training_epochs}')
    model.to(device)
    if training_epochs != 0:
        t5.to('cpu')
        clip.to('cpu')
        ae.to('cpu')
        torch.set_grad_enabled(True)
        inp_optim["img"] = z0
        _, info, _, _, trainable_noise_list = denoise_with_noise_optim(model,**inp_optim,token_ids=token_ids,source_mask=source_mask,training_steps=1,training_epochs=training_epochs,learning_rate=0.01,seed=seed,noise_scale=noise_scale,timesteps=timesteps,info=info,guidance=denoise_guidance)
        z_optim = trainable_noise_list[0]
        info = info
    else:
        z_optim = add_masked_noise_to_z(z0,source_mask,width,height,seed=seed,noise_scale=noise_scale)
        trainable_noise_list = None

    # denoise (editing)
    inp_target["img"] = z_optim
    timesteps = get_schedule(inversion_num_steps, inp_target["img"].shape[1], shift=True)
    model.eval()
    torch.set_grad_enabled(False)
    model.to(device)
    x, _, _, _ = denoise(model, **inp_target, timesteps=timesteps, guidance=denoise_guidance, inverse=False, info=info, trainable_noise_list = trainable_noise_list)
    
     # decode latents to pixel space
    batch_x = unpack(x.float(), width,height)
    ae.to(device)
    for x in batch_x:
        x = x.unsqueeze(0)
        

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
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
        if save:
            output_path = os.path.join(output_dir,f'{target_object}_{injection_num_steps:02d}_{inversion_num_steps}_seed_{seed}_epoch_{training_epochs:03d}_scale_{noise_scale:.2f}.png')
            img.save(output_path, exif=exif_data, quality=95, subsampling=0)
            masked_image.save(output_path.replace(target_object,f'{target_object}_masked'))
            binary_mask = np.where(mask != 0, 255, 0).astype(np.uint8)
            Image.fromarray(binary_mask, mode="L").save(output_path.replace(target_object,f'{target_object}_mask'))
        t1 = time.perf_counter()  
        print(f"Done in {t1 - t0:.1f}s.", f'Saving {output_path} .' if save else 'No saving files.')
    t5.to(device)
    clip.to(device)
    torch.cuda.empty_cache()
    return img

def get_v_src_masks(mask,width,height,device,txt_length=512):
    # resize mask to token size
    mask = (mask > 127).astype(np.uint8)
    mask = mask * 255
    pil_mask = Image.fromarray(mask)
    pil_mask = pil_mask.resize((math.ceil(height/16), math.ceil(width/16)), Image.Resampling.LANCZOS)

    mask = transform(pil_mask)
    mask = mask.flatten().to(device)

    s_mask = mask.view(1, 1, -1, 1)
    s_mask = s_mask.to(torch.bfloat16)
    v_mask = torch.cat([torch.ones(txt_length).to(device),mask])
    v_mask = v_mask.view(1, 1, -1, 1)
    v_mask = v_mask.to(torch.bfloat16)
    return v_mask,s_mask

def create_demo(model_name: str):
    is_schnell = model_name == "flux-schnell"
    
    title = r"""
        <h1 align="center">🎨 LORE Image Editing </h1>
        """
        
    description = r"""
        <b>Official 🤗 Gradio demo</b> <br>
        <b>LORE: Latent Optimization for Precise Semantic Control in Rectified Flow-based Image Editing.</b><br>
        <b>Here are editing steps:</b> <br>
        1️⃣ Upload your source image. <br>
        2️⃣ Fill in your source prompt and use the brush tool to draw your mask. (on layer 1) <br>
        3️⃣ Fill in your target prompt, target object and its index in target prompt (index start from 0). <br>
        4️⃣ Adjust the hyperparameters. <br>
        5️⃣ Click the "Edit" button to generate your edited image! <br>

        🎨 [<b>Examples</b>] Click our examples below, draw your mask and click the "Edit" button. <br>
        🔔 [<b>Note</b>] Due to limited resources in spaces, this demo may only support optimization steps = 1. <br>
        🔔 [<b>Note</b>] Due to limited resources in spaces, you may need to resize large images <= 480. <br>
        If you need high resolution for better quality, go to https://github.com/oyly16/LORE for more usage with your own resource. <br>
        """
    article = r"""
    https://github.com/oyly16/LORE 
    """
    
    with gr.Blocks() as demo:
        gr.HTML(title)
        gr.Markdown(description)
        
        with gr.Row():
            with gr.Column():
                src_prompt = gr.Textbox(label="Source Prompt", value='' )
                inversion_num_steps = gr.Slider(1, 50, 15, step=1, label="Number of inversion/denoise steps")
                injection_num_steps = gr.Slider(1, 50, 12, step=1, label="Number of masked value injection steps")
                target_prompt = gr.Textbox(label="Target Prompt", value='' )
                target_object = gr.Textbox(label="Target Object", value='' )
                target_object_index = gr.Textbox(label="Target Object Index (start index from 0 in target prompt)", value='' )
                brush_canvas = gr.ImageEditor(label="Brush Canvas",
                                                sources=('upload'), 
                                                brush=gr.Brush(colors=["#ff0000"],color_mode='fixed',default_color="#ff0000"),
                                                interactive=True,
                                                transforms=[],
                                                container=True,
                                                format='png',scale=1)
                
                edit_btn = gr.Button("edit")
                
                
            with gr.Column():
                with gr.Accordion("Advanced Options", open=True):

                    training_epochs = gr.Slider(0, 30, 10, step=1, label="LORE optimization steps")
                    inversion_guidance = gr.Slider(1.0, 10.0, 1.0, step=0.1, label="Inversion Guidance", interactive=not is_schnell)
                    denoise_guidance = gr.Slider(1.0, 10.0, 2.0, step=0.1, label="Denoise Guidance", interactive=not is_schnell)
                    noise_scale = gr.Slider(0.0, 1.0, 0.9, step=0.1, label="renoise scale")
                    seed = gr.Textbox('0', label="Seed (-1 for random)", visible=True)
                    resize_longside = gr.Textbox('480', label="Resize (only if input lager than this)(-1 for no resize)", visible=True)
                
                output_image = gr.Image(label="Generated Image")
                gr.Markdown(article)
        edit_btn.click(
            fn=edit,
            inputs=[brush_canvas,src_prompt,inversion_guidance,
                    target_prompt, target_object,target_object_index,
                    inversion_num_steps, injection_num_steps, 
                    training_epochs, 
                    denoise_guidance,noise_scale,seed,resize_longside,
                    ],
            outputs=[output_image]
        )
        gr.Examples(
            examples=[
                ["examples/woman.png", "a young woman", 15, 12, "a young woman with a necklace", "necklace", "5", 10, 0.9, "3", "-1"],
                ["examples/car.png", "a taxi in a neon-lit street", 30, 24, "a race car in a neon-lit street", "race car", "1", 5, 0.1, "2388791121", "-1"],
                ["examples/cup.png", "a cup on a wooden table", 10, 8, "a wooden table", "table", "2", 2, 0, "0", "-1"],
            ],
            inputs=[
                brush_canvas,
                src_prompt,
                inversion_num_steps,
                injection_num_steps,
                target_prompt,
                target_object,
                target_object_index,
                training_epochs,
                noise_scale,
                seed,
                resize_longside
            ],
            label="Examples (Click to load)"
        )

    return demo

demo = create_demo("flux-dev")    
demo.launch(share=True)
