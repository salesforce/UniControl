'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
 * Modified from ControlNet repo: https://github.com/lllyasviel/ControlNet
 * Copyright (c) 2023 Lvmin Zhang and Maneesh Agrawala
'''

from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_unicontrol_hacked import DDIMSampler
from annotator.outpainting import Outpainter

model_outpainting = Outpainter()
def outpainting(img, res, rand_h, rand_w):
    img = resize_image(HWC3(img), res)   
    result = model_outpainting(img, rand_h, rand_w)
    return result

model = create_model('./models/cldm_v15_unicontrol.yaml').cpu()
model.load_state_dict(load_state_dict('./ckpts/unicontrol.ckpt', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

task_to_name = {'hed': 'control_hed', 'canny': 'control_canny', 'seg': 'control_seg', 'segbase': 'control_seg', 'depth': 'control_depth', 'normal': 'control_normal', 'openpose': 'control_openpose', 'bbox': 'control_bbox', 'grayscale': 'control_grayscale', 'outpainting': 'control_outpainting', 'hedsketch': 'control_hedsketch'}

name_to_instruction = {"control_hed": "hed edge to image", "control_canny": "canny edge to image", "control_seg": "segmentation map to image", "control_depth": "depth map to image", "control_normal": "normal surface map to image", "control_img": "image editing", "control_openpose": "human pose skeleton to image", "control_hedsketch": "sketch to image", "control_bbox": "bounding box to image", "control_outpainting": "image outpainting"}


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, h_ratio, w_ratio):
    with torch.no_grad():
        input_image = HWC3(input_image)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        detected_map = outpainting(input_image, image_resolution, h_ratio, w_ratio)        
        

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
            
        task = 'outpainting'    
        task_dic = {}
        task_dic['name'] = task_to_name[task]
        task_instruction = name_to_instruction[task_dic['name']]
        task_dic['feature'] = model.get_learned_conditioning(task_instruction)[:,:1,:]
        
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], "task": task_dic}
        
#         cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

#         model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## UniControl Stable Diffusion with Image Outpainting")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                h_ratio = gr.Slider(label="Height Masking Ratio", minimum=20, maximum=80, value=50, step=1)
                w_ratio = gr.Slider(label="Width Masking Ratio", minimum=20, maximum=80, value=50, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt", value='')
#                 n_prompt = gr.Textbox(label="Negative Prompt",
#                                       value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, h_ratio, w_ratio]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(share=True, server_name='0.0.0.0')
