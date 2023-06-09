'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
'''


import json
import cv2
import numpy as np

from torch.utils.data import Dataset
from annotator.util import resize_image, HWC3
import random

class MyDataset(Dataset):
    def __init__(self, path_json, path_meta, task ):
        self.data = []
        with open(path_json, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.path_meta = path_meta
        if task == 'hed':
            self.key_prompt = 'control_hed'
        elif task == 'canny':
            self.key_prompt = 'control_canny'
        elif task == 'seg' or task == 'segbase':
            self.key_prompt = 'control_seg'
        elif task == 'depth':
            self.key_prompt = 'control_depth'
        elif task == 'normal':
            self.key_prompt = 'control_normal'
        elif task == 'openpose':
            self.key_prompt = 'control_openpose'
        elif task == 'hedsketch':
            self.key_prompt = 'control_hedsketch'
        elif task == 'bbox':
            self.key_prompt = 'control_bbox'
        elif task == 'outpainting':
            self.key_prompt = 'control_outpainting' 
        elif task == 'inpainting':
            self.key_prompt = 'control_inpainting'
        elif task == 'blur':
            self.key_prompt = 'control_blur'
        elif task == 'grayscale':
            self.key_prompt = 'control_grayscale'
        else:
            print('TASK NOT MATCH')
        self.key_prompt_model = self.key_prompt   
        self.resolution = 512
        
        self.task_to_instruction = {"control_hed": "hed edge to image", "control_canny": "canny edge to image", "control_seg": "segmentation map to image", "control_depth": "depth map to image", "control_normal": "normal surface map to image", "control_img": "image editing", "control_openpose": "human pose skeleton to image", "control_hedsketch": "sketch to image", "control_bbox": "bounding box to image", "control_outpainting": "image outpainting", "control_grayscale": "gray image to color image", "control_blur": "deblur image to clean image", "control_inpainting": "image inpainting"}
        
    def __len__(self):
        return len(self.data)
    
    def resize_image_control(self, control_image, resolution):
        H, W, C = control_image.shape
        if W >= H:
            crop = H
            crop_l = random.randint(0, W-crop) # 2nd value is inclusive
            crop_r = crop_l + crop
            crop_t = 0
            crop_b = H
        else:
            crop = W
            crop_t = random.randint(0, H-crop) # 2nd value is inclusive
            crop_b = crop_t + crop
            crop_l = 0
            crop_r = W
        control_image = control_image[ crop_t: crop_b, crop_l:crop_r]
        H = float(H)
        W = float(W)
        k = float(resolution) / min(H, W)
        img = cv2.resize(control_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return img, [crop_t/H, crop_b/H, crop_l/W, crop_r/W]

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item[self.key_prompt]
        print(self.path_meta + source_filename)
        source_img = cv2.imread(self.path_meta + source_filename)
        target_filename = item['source']
        if "./" == target_filename[0:2]:
            target_filename = target_filename[2:]
            
        prompt = item['prompt']

        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        source_img,  _ = self.resize_image_control(source_img, self.resolution)

        # Normalize source images to [0, 1].
        source_img = source_img.astype(np.float32) / 255.0

        return dict(txt=prompt, hint=source_img, task=self.key_prompt_model, instruction=self.task_to_instruction[self.key_prompt_model])

