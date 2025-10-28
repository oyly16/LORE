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
from flux.sampling_lore import denoise, get_schedule, prepare, unpack, get_v_mask, add_masked_noise_to_z,get_mask_one_tensor, denoise_with_noise_optim,prepare_tokens
from flux.util_lore import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5)

def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image
from torchvision import transforms
transform = transforms.ToTensor()

class FluxEditor_lore_demo:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.offload = args.offload

        self.name = args.name
        self.is_schnell = args.name == "flux-schnell"
        self.resize_longside = args.resize
        self.save = args.save

        self.output_dir = 'outputs_gradio'

        self.t5 = load_t5(self.device, max_length=256 if self.name == "flux-schnell" else 512)
        self.clip = load_clip(self.device)
        self.model = load_flow_model(args.name, device=self.device)
        self.ae = load_ae(self.name, device="cpu" if self.offload else self.device)

        self.t5.eval()
        self.clip.eval()
        self.ae.eval()
        self.info = {}
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.encoder.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False  # freeze the model
        for param in self.t5.parameters():
            param.requires_grad = False  # freeze the model
        for param in self.clip.parameters():
            param.requires_grad = False  # freeze the model
        for param in self.ae.parameters():
            param.requires_grad = False  # freeze the model
        
    def resize_image(self,image):
        pil_image = Image.fromarray(image)
        h, w = pil_image.size[1], pil_image.size[0]
        if h <= self.resize_longside and w <= self.resize_longside:
            return image
    
        if h >= w:
            new_h = self.resize_longside 
            new_w = int(w * self.resize_longside  / h)
        else:
            new_w = self.resize_longside 
            new_h = int(h * self.resize_longside  / w)
    
        resized_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
        return np.array(resized_image)
        
    def resize_mask(self,mask,height,width):
        pil_mask = Image.fromarray(mask.astype(np.uint8))  # ensure it's 8-bit for PIL
        resized_pil = pil_mask.resize((width, height), Image.NEAREST)  # width first!
        return np.array(resized_pil)
   
    def inverse(self, brush_canvas,src_prompt, 
                inversion_num_steps, injection_num_steps, 
                inversion_guidance,
             ):
        print(f"Inversing {src_prompt}, guidance {inversion_guidance}, inje/step {injection_num_steps}/{inversion_num_steps}")
        self.z0 = None
        self.zt = None
        torch.cuda.empty_cache()
        if self.info:
            del self.info        
        self.info = {'src_p':src_prompt}
        
        rgba_init_image = brush_canvas["background"]
        init_image = rgba_init_image[:,:,:3]
        
            
        if self.resize_longside != -1:
            init_image = self.resize_image(init_image)
        shape = init_image.shape

        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
    
        init_image = init_image[:new_h, :new_w, :]
        width, height = init_image.shape[0], init_image.shape[1]
        self.init_image = encode(init_image, self.device, self.ae)

        if self.save:
            ori_output_path = os.path.join(self.output_dir,f'{src_prompt[:20]}_ori.png')
            Image.fromarray(init_image,'RGB').save(ori_output_path)
        
        t0 = time.perf_counter()

        self.info['feature'] = {}
        self.info['inject_step'] = injection_num_steps
        self.info['wh'] = (width, height)
        
        torch.cuda.empty_cache()

        inp = prepare(self.t5, self.clip, self.init_image, prompt=src_prompt)
        timesteps = get_schedule(inversion_num_steps, inp["img"].shape[1], shift=True)
        self.info['x_ori'] = inp["img"].clone()
    
        # inversion initial noise
        torch.set_grad_enabled(False)
        z, info, _, _ = denoise(self.model, **inp, timesteps=timesteps, guidance=inversion_guidance, inverse=True, info=self.info)
        self.z0 = z
        self.info = info
            
        t1 = time.perf_counter()
        print(f"inversion Done in {t1 - t0:.1f}s.")
        return init_image

    def edit(self, brush_canvas, source_prompt, inversion_guidance,
                target_prompt, target_object,target_object_index,
                inversion_num_steps, injection_num_steps, 
                training_epochs, 
                denoise_guidance,noise_scale,seed,
             ):
        
        torch.cuda.empty_cache()
        if 'src_p' not in self.info or self.info['src_p'] != source_prompt:
            print('src prompt changed. inverse again')
            self.inverse(brush_canvas,source_prompt, 
                inversion_num_steps, injection_num_steps, 
                inversion_guidance)
        
        rgba_init_image = brush_canvas["background"]
        rgba_mask = brush_canvas["layers"][0]
        init_image = rgba_init_image[:,:,:3]
        if self.resize_longside != -1:
            init_image = self.resize_image(init_image)
        width, height = self.info['wh']
        init_image = init_image[:width, :height, :]
        #rgba_init_image = rgba_init_image[:height, :width, :]
        
        if self.resize_longside != -1:
            mask = self.resize_mask(rgba_mask[:,:,3],height,width)
        else:            
            mask = rgba_mask[:width, :height, 3]
        mask = mask.astype(int)
        
        rgba_mask[:,:,3] = rgba_mask[:,:,3]//2
        masked_image = Image.alpha_composite(Image.fromarray(rgba_init_image, 'RGBA'), Image.fromarray(rgba_mask, 'RGBA'))
        masked_image = masked_image.resize((height, width), Image.LANCZOS)
            

        # prepare source mask and vmask
        inp_optim = prepare(self.t5, self.clip, self.init_image, prompt=target_prompt)
        inp_target = prepare(self.t5, self.clip, self.init_image, prompt=target_prompt)
        v_mask,source_mask = self.get_v_src_masks(mask,width,height,self.device)
        self.info['change_v'] = 2 # v_mask
        self.info['v_mask'] = v_mask
        self.info['source_mask'] = source_mask
        self.info['inject_step'] = injection_num_steps
        timesteps = get_schedule(inversion_num_steps, inp_optim["img"].shape[1], shift=True)
        seed = int(seed)
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()

        # prepare token_ids
        token_ids=[]
        replacements = [[None,target_object,-1,int(target_object_index)]]
        src_dif_ids,tgt_dif_ids = prepare_tokens(self.t5, source_prompt, target_prompt, replacements,True)
        for t_ids in tgt_dif_ids:
            token_ids.append([t_ids,True,1])
        print('token_ids',token_ids)
        # do latent optim
        self.t5.to('cpu')
        self.clip.to('cpu')
        t0 = time.perf_counter() 
        print(f'optimizing & editing noise, {target_prompt} with seed {seed}, noise_scale {noise_scale}, training_epochs {training_epochs}')
        if training_epochs != 0:
            torch.set_grad_enabled(True)
            inp_optim["img"] = self.z0
            _, info, _, _, trainable_noise_list = denoise_with_noise_optim(self.model,**inp_optim,token_ids=token_ids,source_mask=source_mask,training_steps=1,training_epochs=training_epochs,learning_rate=0.01,seed=seed,noise_scale=noise_scale,timesteps=timesteps,info=self.info,guidance=denoise_guidance)
            z_optim = trainable_noise_list[0]
            self.info = info
        else:
            z_optim = add_masked_noise_to_z(self.z0,source_mask,width,height,seed=seed,noise_scale=noise_scale)
            trainable_noise_list = None

        # denoise (editing)
        inp_target["img"] = z_optim
        timesteps = get_schedule(inversion_num_steps, inp_target["img"].shape[1], shift=True)
        self.model.eval()
        torch.set_grad_enabled(False)
        x, _, _, _ = denoise(self.model, **inp_target, timesteps=timesteps, guidance=denoise_guidance, inverse=False, info=self.info, trainable_noise_list = trainable_noise_list)
        
         # decode latents to pixel space
        batch_x = unpack(x.float(), width,height)
    
        for x in batch_x:
            x = x.unsqueeze(0)
            
    
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                x = self.ae.decode(x)
    
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
            if self.save:
                output_path = os.path.join(self.output_dir,f'{target_object}_{injection_num_steps:02d}_{inversion_num_steps}_seed_{seed}_epoch_{training_epochs:03d}_scale_{noise_scale:.2f}.png')
                img.save(output_path, exif=exif_data, quality=95, subsampling=0)
                masked_image.save(output_path.replace(target_object,f'{target_object}_masked'))
                binary_mask = np.where(mask != 0, 255, 0).astype(np.uint8)
                Image.fromarray(binary_mask, mode="L").save(output_path.replace(target_object,f'{target_object}_mask'))
            t1 = time.perf_counter()  
            print(f"Done in {t1 - t0:.1f}s.", f'Saving {output_path} .' if self.save else 'No saving files.')
        self.t5.to(self.device)
        self.clip.to(self.device)
        return img

    def encode(self,init_image, torch_device):
        init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
        init_image = init_image.unsqueeze(0) 
        init_image = init_image.to(torch_device)
        self.ae.encoder.to(torch_device)
        
        init_image = self.ae.encode(init_image).to(torch.bfloat16)
        return init_image

    def get_v_src_masks(self,mask,width,height,device,txt_length=512):
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
    editor = FluxEditor_lore_demo(args)
    is_schnell = model_name == "flux-schnell"
    
    title = r"""
        <h1 align="center">🎨 LORE Image Editing </h1>
        """
        
    description = r"""
        <b>Official 🤗 Gradio demo</b> <br>
        <b>LORE: Latent Optimization for Precise Semantic Control in Rectified Flow-based Image Editing.</b><br>
        <b>Here are editing steps:</b> <br>
        1️⃣ Upload your source image. <br>
        2️⃣ Fill in your source prompt and click the "Inverse" button to perform image inversion. <br>
        3️⃣ Use the brush tool to draw your mask. (on layer 1) <br>
        4️⃣ Fill in your target prompt, then adjust the hyperparameters. <br>
        5️⃣ Click the "Edit" button to generate your edited image! <br>
        6️⃣ If source image and prompt are not changed, you can click 'Edit' for next generation. <br>
        
        🔔 [<b>Note</b>] Due to limited resources, we will resize image to <=800 longside. <br>
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
                
                inv_btn = gr.Button("inverse")
                edit_btn = gr.Button("edit")
                
                
            with gr.Column():
                with gr.Accordion("Advanced Options", open=True):

                    training_epochs = gr.Slider(0, 30, 10, step=1, label="Number of LORE training epochs")
                    inversion_guidance = gr.Slider(1.0, 10.0, 1.0, step=0.1, label="inversion Guidance", interactive=not is_schnell)
                    denoise_guidance = gr.Slider(1.0, 10.0, 2.0, step=0.1, label="denoise Guidance", interactive=not is_schnell)
                    noise_scale = gr.Slider(0.0, 1.0, 0.9, step=0.1, label="renoise scale")
                    seed = gr.Textbox('0', label="Seed (-1 for random)", visible=True)

                
                output_image = gr.Image(label="Generated Image")
                gr.Markdown(article)
        inv_btn.click(
            fn=editor.inverse,
            inputs=[brush_canvas,src_prompt, 
                    inversion_num_steps, injection_num_steps, 
                    inversion_guidance,
                    ],
            outputs=[output_image]
        )
        edit_btn.click(
            fn=editor.edit,
            inputs=[brush_canvas,src_prompt,inversion_guidance,
                    target_prompt, target_object,target_object_index,
                    inversion_num_steps, injection_num_steps, 
                    training_epochs, 
                    denoise_guidance,noise_scale,seed,
                    ],
            outputs=[output_image]
        )
        gr.Examples(
            examples=[
                ["examples/woman.png", "a young woman", 15, 12, "a young woman with a necklace", "necklace", "5", 10, 0.9, "3"],
                ["examples/car.png", "a taxi in a neon-lit street", 30, 24, "a race car in a neon-lit street", "race car", "1", 5, 0.1, "2388791121"],
                ["examples/cup.png", "a cup on a wooden table", 10, 8, "a wooden table", "table", "2", 2, 0, "0"],
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
            ],
            label="Examples (Click to load)"
        )

    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument("--name", type=str, default="flux-dev", choices=list(configs.keys()), help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--share", action="store_true", help="Create a public link to your demo")
    parser.add_argument("--resize", type=int, default=800, help="Resize long side")
    parser.add_argument("--save", action="store_true", help="Save edits to local files")
    parser.add_argument("--port", type=int, default=41032)
    args = parser.parse_args()

    demo = create_demo(args.name)
    
    demo.launch(server_name='0.0.0.0', share=args.share, server_port=args.port)
