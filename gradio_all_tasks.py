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
from annotator.uniformer_base import UniformerDetector
from annotator.hed import HEDdetector
from annotator.canny import CannyDetector
from annotator.midas import MidasDetector
from annotator.outpainting import Outpainter
from annotator.openpose import OpenposeDetector
import cvlib as cv

from cldm.model import create_model, load_state_dict
from cldm.ddim_unicontrol_hacked import DDIMSampler
# from cldm.ddim_hacked import DDIMSampler
import pdb

apply_uniformer = UniformerDetector()
apply_midas = MidasDetector()
apply_canny = CannyDetector()
apply_hed = HEDdetector()
model_outpainting = Outpainter()
apply_openpose = OpenposeDetector()

def midas(img, res):
    img = resize_image(HWC3(img), res)
    results = apply_midas(img)
    return results

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


def process_canny(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold, condition_mode):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape
        if condition_mode == True:
            detected_map = apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)
        else:
            detected_map = img

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        task = 'canny'    
        task_dic = {}
        task_dic['name'] = task_to_name[task]
        task_instruction = name_to_instruction[task_dic['name']]
        task_dic['feature'] = model.get_learned_conditioning(task_instruction)[:,:1,:]
        
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], "task": task_dic}
        
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results

def process_hed(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, condition_mode):
    with torch.no_grad():
        input_image = HWC3(input_image)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        if condition_mode == True:
            detected_map = apply_hed(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
        else:
            detected_map = img
            
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
            
        task = 'hed'    
        task_dic = {}
        task_dic['name'] = task_to_name[task]
        task_instruction = name_to_instruction[task_dic['name']]
        task_dic['feature'] = model.get_learned_conditioning(task_instruction)[:,:1,:]
        
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], "task": task_dic}
        
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

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


def process_depth(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, condition_mode):
    with torch.no_grad():
#         input_image = HWC3(input_image)
# #         detected_map, _ = midas(input_image, image_resolution)
#         detected_map, _ = apply_midas(resize_image(input_image, image_resolution))
# #         detected_map = HWC3(detected_map)
#         img = resize_image(input_image, image_resolution)
#         H, W, C = img.shape

# #         detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        input_image = HWC3(input_image)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        if condition_mode == True:
            detected_map, _ = apply_midas(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
        else:
            detected_map = img

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        task = 'depth'    
        task_dic = {}
        task_dic['name'] = task_to_name[task]
        task_instruction = name_to_instruction[task_dic['name']]
        task_dic['feature'] = model.get_learned_conditioning(task_instruction)[:,:1,:]
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], "task": task_dic}

        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

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

def process_normal(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, condition_mode):
    with torch.no_grad():
#         input_image = HWC3(input_image)
#         _, detected_map = apply_midas(resize_image(input_image, image_resolution))
#         detected_map = HWC3(detected_map)
# #         img = resize_image(input_image, image_resolution)
#         H, W, C = input_image.shape

#         detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        input_image = HWC3(input_image)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        if condition_mode == True:
            _, detected_map = apply_midas(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
        else:
            detected_map = img

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        task = 'normal'    
        task_dic = {}
        task_dic['name'] = task_to_name[task]
        task_instruction = name_to_instruction[task_dic['name']]
        task_dic['feature'] = model.get_learned_conditioning(task_instruction)[:,:1,:]
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], "task": task_dic}
        
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

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

def process_pose(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, condition_mode):
    with torch.no_grad():
        input_image = HWC3(input_image)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        if condition_mode == True:
            detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
        else:
            detected_map = img

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        task = 'openpose'    
        task_dic = {}
        task_dic['name'] = task_to_name[task]
        task_instruction = name_to_instruction[task_dic['name']]
        task_dic['feature'] = model.get_learned_conditioning(task_instruction)[:,:1,:]
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], "task": task_dic}

        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
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

def process_seg(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, condition_mode):
    with torch.no_grad():
        input_image = HWC3(input_image)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        
        if condition_mode == True:
            detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
        else:
            detected_map = img
        
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        task = 'seg'    
        task_dic = {}
        task_dic['name'] = task_to_name[task]
        task_instruction = name_to_instruction[task_dic['name']]
        task_dic['feature'] = model.get_learned_conditioning(task_instruction)[:,:1,:]
        
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], "task": task_dic}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
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

color_dict = {
    'background': (0, 0, 100),
    'person': (255, 0, 0),
    'bicycle': (0, 255, 0),
    'car': (0, 0, 255),
    'motorcycle': (255, 255, 0),
    'airplane': (255, 0, 255),
    'bus': (0, 255, 255),
    'train': (128, 128, 0),
    'truck': (128, 0, 128),
    'boat': (0, 128, 128),
    'traffic light': (128, 128, 128),
    'fire hydrant': (64, 0, 0),
    'stop sign': (0, 64, 0),
    'parking meter': (0, 0, 64),
    'bench': (64, 64, 0),
    'bird': (64, 0, 64),
    'cat': (0, 64, 64),
    'dog': (192, 192, 192),
    'horse': (32, 32, 32),
    'sheep': (96, 96, 96),
    'cow': (160, 160, 160),
    'elephant': (224, 224, 224),
    'bear': (32, 0, 0),
    'zebra': (0, 32, 0),
    'giraffe': (0, 0, 32),
    'backpack': (32, 32, 0),
    'umbrella': (32, 0, 32),
    'handbag': (0, 32, 32),
    'tie': (96, 0, 0),
    'suitcase': (0, 96, 0),
    'frisbee': (0, 0, 96),
    'skis': (96, 96, 0),
    'snowboard': (96, 0, 96),
    'sports ball': (0, 96, 96),
    'kite': (160, 0, 0),
    'baseball bat': (0, 160, 0),
    'baseball glove': (0, 0, 160),
    'skateboard': (160, 160, 0),
    'surfboard': (160, 0, 160),
    'tennis racket': (0, 160, 160),
    'bottle': (224, 0, 0),
    'wine glass': (0, 224, 0),
    'cup': (0, 0, 224),
    'fork': (224, 224, 0),
    'knife': (224, 0, 224),
    'spoon': (0, 224, 224),
    'bowl': (64, 64, 64),
    'banana': (128, 64, 64),
    'apple': (64, 128, 64),
    'sandwich': (64, 64, 128),
    'orange': (128, 128, 64),
    'broccoli': (128, 64, 128),
    'carrot': (64, 128, 128),
    'hot dog': (192, 64, 64),
    'pizza': (64, 192, 64),
    'donut': (64, 64, 192),
    'cake': (192, 192, 64),
    'chair': (192, 64, 192),
    'couch': (64, 192, 192),
    'potted plant': (96, 32, 32),
    'bed': (32, 96, 32),
    'dining table': (32, 32, 96),
    'toilet': (96, 96, 32),
    'tv': (96, 32, 96),
    'laptop': (32, 96, 96),
    'mouse': (160, 32, 32),
    'remote': (32, 160, 32),
    'keyboard': (32, 32, 160),
    'cell phone': (160, 160, 32),
    'microwave': (160, 32, 160),
    'oven': (32, 160, 160),
    'toaster': (224, 32, 32),
    'sink': (32, 224, 32),
    'refrigerator': (32, 32, 224),
    'book': (224, 224, 32),
    'clock': (224, 32, 224),
    'vase': (32, 224, 224),
    'scissors': (64, 96, 96),
    'teddy bear': (96, 64, 96),
    'hair drier': (96, 96, 64),
    'toothbrush': (160, 96, 96)
}

def process_bbox(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, confidence,nms_thresh, condition_mode):
    with torch.no_grad():
        input_image = HWC3(input_image)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        
        if condition_mode == True:
            bbox, label, conf = cv.detect_common_objects(input_image, confidence=confidence, nms_thresh=nms_thresh)
            mask = np.zeros((input_image.shape), np.uint8)
            if len(bbox) > 0:
                order_area = np.zeros(len(bbox))
            #     order_final = np.arange(len(bbox))
                area_all = 0
                for idx_mask, box in enumerate(bbox):
                    x_1, y_1, x_2, y_2 = box

                    x_1 = 0 if x_1 < 0 else x_1
                    y_1 = 0 if y_1 < 0 else y_1
                    x_2 = input_image.shape[1] if x_2 < 0 else x_2
                    y_2 = input_image.shape[0] if y_2 < 0 else y_2

                    area = (x_2 - x_1) * (y_2 - y_1)
                    order_area[idx_mask] = area
                    area_all += area
                ordered_area = np.argsort(-order_area)

                for idx_mask in ordered_area:
                    box = bbox[idx_mask]
                    x_1, y_1, x_2, y_2 = box
                    x_1 = 0 if x_1 < 0 else x_1
                    y_1 = 0 if y_1 < 0 else y_1
                    x_2 = input_image.shape[1] if x_2 < 0 else x_2
                    y_2 = input_image.shape[0] if y_2 < 0 else y_2

                    mask[y_1:y_2, x_1:x_2, :] = color_dict[label[idx_mask]]
            detected_map = mask
        else:
            detected_map = img

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
            
        task = 'bbox'    
        task_dic = {}
        task_dic['name'] = task_to_name[task]
        task_instruction = name_to_instruction[task_dic['name']]
        task_dic['feature'] = model.get_learned_conditioning(task_instruction)[:,:1,:]
        
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], "task": task_dic}
        
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

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

def process_outpainting(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, h_ratio, w_ratio, condition_mode):
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
        
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
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

def process_sketch(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, condition_mode):
    with torch.no_grad():
        input_image = HWC3(input_image)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        
        if condition_mode == True:
            detected_map = apply_hed(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)

            # sketch the hed image
            retry = 0
            cnt = 0
            while retry == 0:
                threshold_value = np.random.randint(110, 160)
                kernel_size = 3
                alpha = 1.5
                beta = 50
                binary_image = cv2.threshold(detected_map, threshold_value, 255, cv2.THRESH_BINARY)[1]
                inverted_image = cv2.bitwise_not(binary_image)
                smoothed_image = cv2.GaussianBlur(inverted_image, (kernel_size, kernel_size), 0)
                sketch_image = cv2.convertScaleAbs(smoothed_image, alpha=alpha, beta=beta)
                if np.sum(sketch_image < 5) > 0.005 * sketch_image.shape[0] * sketch_image.shape[1] or cnt == 5:
                    retry = 1
                else:
                    cnt += 1
            detected_map = sketch_image
        else:
            detected_map = img
        

        detected_map =  cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR) 
        
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
            
        task = 'hedsketch'    
        task_dic = {}
        task_dic['name'] = task_to_name[task]
        task_instruction = name_to_instruction[task_dic['name']]
        task_dic['feature'] = model.get_learned_conditioning(task_instruction)[:,:1,:]
        
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], "task": task_dic}
        
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
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

demo = gr.Blocks()
with demo:
    gr.Markdown("UniControl Stable Diffusion Demo")
    with gr.Tabs():
        with gr.TabItem("Canny"):
            with gr.Row():
                gr.Markdown("## UniControl Stable Diffusion with Canny Edge Maps")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(source='upload', type="numpy")
                    prompt = gr.Textbox(label="Prompt")
                    run_button = gr.Button(label="Run")
                    with gr.Accordion("Advanced options", open=False):
                        num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=3, step=1)
                        image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                        strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        condition_mode = gr.Checkbox(label='Condition Extraction', value=True)
                        guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                        low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=40, step=1)
                        high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                        ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                        scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        eta = gr.Number(label="eta (DDIM)", value=0.0)
                        a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed, bright')
                        n_prompt = gr.Textbox(label="Negative Prompt", value='')
                with gr.Column():
                    result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold, condition_mode]
            run_button.click(fn=process_canny, inputs=ips, outputs=[result_gallery])
    
        with gr.TabItem("HED"):
            with gr.Row():
                gr.Markdown("## UniControl Stable Diffusion with HED Maps")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(source='upload', type="numpy")
                    prompt = gr.Textbox(label="Prompt")
                    run_button = gr.Button(label="Run")
                    with gr.Accordion("Advanced options", open=False):
                        num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=3, step=1)
                        image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                        strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        condition_mode = gr.Checkbox(label='Condition Extraction', value=True)
                        guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                        detect_resolution = gr.Slider(label="HED Resolution", minimum=128, maximum=1024, value=512, step=1)
                        ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                        scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        eta = gr.Number(label="eta (DDIM)", value=0.0)
                        a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed, bright')
                        n_prompt = gr.Textbox(label="Negative Prompt", value='')
                with gr.Column():
                    result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, condition_mode]
            run_button.click(fn=process_hed, inputs=ips, outputs=[result_gallery])
        
        with gr.TabItem("Sketch"):
            with gr.Row():
                gr.Markdown("## UniControl Stable Diffusion with Sketch Maps")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(source='upload', type="numpy")
                    prompt = gr.Textbox(label="Prompt")
                    run_button = gr.Button(label="Run")
                    with gr.Accordion("Advanced options", open=False):
                        num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=3, step=1)
                        image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                        strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        condition_mode = gr.Checkbox(label='Condition Extraction', value=True)
                        guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                        detect_resolution = gr.Slider(label="HED Resolution", minimum=128, maximum=1024, value=512, step=1)
                        ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                        scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        eta = gr.Number(label="eta (DDIM)", value=0.0)
                        a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                        n_prompt = gr.Textbox(label="Negative Prompt", value='')
                with gr.Column():
                    result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, condition_mode]
            run_button.click(fn=process_sketch, inputs=ips, outputs=[result_gallery])
            
        with gr.TabItem("Depth"):
            with gr.Row():
                gr.Markdown("## UniControl Stable Diffusion with Depth Maps")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(source='upload', type="numpy")
                    prompt = gr.Textbox(label="Prompt")
                    run_button = gr.Button(label="Run")
                    with gr.Accordion("Advanced options", open=False):
                        num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=3, step=1)
                        image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                        strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        condition_mode = gr.Checkbox(label='Condition Extraction', value=True)
                        guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                        detect_resolution = gr.Slider(label="Depth Resolution", minimum=128, maximum=1024, value=384, step=1)
                        ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                        scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        eta = gr.Number(label="eta (DDIM)", value=0.0)
                        a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed, bright')
                        n_prompt = gr.Textbox(label="Negative Prompt", value='')
                with gr.Column():
                    result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, condition_mode]
            run_button.click(fn=process_depth, inputs=ips, outputs=[result_gallery])
    
        with gr.TabItem("Normal"):
            with gr.Row():
                gr.Markdown("## UniControl Stable Diffusion with Normal Surface")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(source='upload', type="numpy")
                    prompt = gr.Textbox(label="Prompt")
                    run_button = gr.Button(label="Run")
                    with gr.Accordion("Advanced options", open=False):
                        num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=3, step=1)
                        image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                        strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        condition_mode = gr.Checkbox(label='Condition Extraction', value=True)
                        guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                        detect_resolution = gr.Slider(label="Depth Resolution", minimum=128, maximum=1024, value=384, step=1)
                        ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                        scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        eta = gr.Number(label="eta (DDIM)", value=0.0)
                        a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed, bright')
                        n_prompt = gr.Textbox(label="Negative Prompt", value='')
                with gr.Column():
                    result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, condition_mode]
            run_button.click(fn=process_normal, inputs=ips, outputs=[result_gallery])
    
        with gr.TabItem("Human Pose"):
            with gr.Row():
                gr.Markdown("## UniControl Stable Diffusion with Human Pose")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(source='upload', type="numpy")
                    prompt = gr.Textbox(label="Prompt")
                    run_button = gr.Button(label="Run")
                    with gr.Accordion("Advanced options", open=False):
                        num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=3, step=1)
                        image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                        strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        condition_mode = gr.Checkbox(label='Condition Extraction', value=True)
                        guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                        detect_resolution = gr.Slider(label="OpenPose Resolution", minimum=128, maximum=1024, value=512, step=1)
                        ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                        scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        eta = gr.Number(label="eta (DDIM)", value=0.0)
                        a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed, bright')
                        n_prompt = gr.Textbox(label="Negative Prompt", value='')
                with gr.Column():
                    result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, condition_mode]
            run_button.click(fn=process_pose, inputs=ips, outputs=[result_gallery])
    
        with gr.TabItem("Segmentation"):
            with gr.Row():
                gr.Markdown("## UniControl Stable Diffusion with Segmentation Maps (ADE20K)")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(source='upload', type="numpy")
                    prompt = gr.Textbox(label="Prompt")
                    run_button = gr.Button(label="Run")
                    with gr.Accordion("Advanced options", open=False):
                        num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=3, step=1)
                        image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                        strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        condition_mode = gr.Checkbox(label='Condition Extraction', value=True)
                        guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                        detect_resolution = gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
                        ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                        scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        eta = gr.Number(label="eta (DDIM)", value=0.0)
                        a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed, bright')
                        n_prompt = gr.Textbox(label="Negative Prompt",value='')
                with gr.Column():
                    result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, condition_mode]
            run_button.click(fn=process_seg, inputs=ips, outputs=[result_gallery])
            
        with gr.TabItem("Bbox"):
            with gr.Row():
                gr.Markdown("## UniControl Stable Diffusion with Object Bounding Boxes (MS-COCO)")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(source='upload', type="numpy")
                    prompt = gr.Textbox(label="Prompt")
                    run_button = gr.Button(label="Run")
                    with gr.Accordion("Advanced options", open=False):
                        num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=3, step=1)
                        image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                        strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        condition_mode = gr.Checkbox(label='Condition Extraction', value=True)
                        guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                        confidence = gr.Slider(label="Confidence of Detection", minimum=0.1, maximum=1.0, value=0.4, step=0.1)
                        nms_thresh = gr.Slider(label="Nms Threshold", minimum=0.1, maximum=1.0, value=0.5, step=0.1)
                        ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                        scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        eta = gr.Number(label="eta (DDIM)", value=0.0)
                        a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed, bright')
                        n_prompt = gr.Textbox(label="Negative Prompt", value='')
                with gr.Column():
                    result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, confidence, nms_thresh, condition_mode]
            run_button.click(fn=process_bbox, inputs=ips, outputs=[result_gallery])
        
        with gr.TabItem("Outpainting"):
            with gr.Row():
                gr.Markdown("## UniControl Stable Diffusion with Image Outpainting")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(source='upload', type="numpy")
                    prompt = gr.Textbox(label="Prompt")
                    run_button = gr.Button(label="Run")
                    with gr.Accordion("Advanced options", open=False):
                        num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=3, step=1)
                        image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                        strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        condition_mode = gr.Checkbox(label='Condition Extraction', value=True)
                        guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                        h_ratio = gr.Slider(label="Height Masking Ratio", minimum=20, maximum=80, value=50, step=1)
                        w_ratio = gr.Slider(label="Width Masking Ratio", minimum=20, maximum=80, value=50, step=1)
                        ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                        scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        eta = gr.Number(label="eta (DDIM)", value=0.0)
                        a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                        n_prompt = gr.Textbox(label="Negative Prompt", value='')
                with gr.Column():
                    result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, h_ratio, w_ratio, condition_mode]
            run_button.click(fn=process_outpainting, inputs=ips, outputs=[result_gallery])

demo.queue(concurrency_count=3)
demo.launch(share=True, server_name='0.0.0.0')
