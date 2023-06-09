'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
'''

import sys

sys.path.append('./')
import json
import cv2
import numpy as np

from torch.utils.data import Dataset
import pdb
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
            
        self.resolution = 512
        self.none_loop = 0
        
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
    
    def resize_image_target(self, target_image, resolution, sizes):
        H, W, C = target_image.shape
        crop_t_rate, crop_b_rate, crop_l_rate, crop_r_rate = sizes[0], sizes[1], sizes[2], sizes[3]
        crop_t, crop_b, crop_l, crop_r = int(crop_t_rate*H), int(crop_b_rate*H), int(crop_l_rate*W), int(crop_r_rate*W)
        target_image = target_image[ crop_t: crop_b, crop_l:crop_r]
        H = float(H)
        W = float(W)
        k = float(resolution) / min(H, W)
        img = cv2.resize(target_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return img
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item[self.key_prompt]
        source_img = cv2.imread(self.path_meta + "/conditions/" + source_filename)
        target_filename = item['source']
        if "./" == target_filename[0:2]:
            target_filename = target_filename[2:]
        target_img = cv2.imread(self.path_meta+ "/images/" + target_filename)
        prompt = item['prompt']
        
        while source_img is None or target_img is None or prompt is None:
            # corner cases
            if idx >= 0 and idx < len(self.data) - 1:
                idx += 1
            elif idx == len(self.data) - 1:
                idx = 0
            item = self.data[idx]
            source_filename = item[self.key_prompt]
            source_img = cv2.imread(self.path_meta + "/conditions/" + source_filename)
            target_filename = item['source']
            if "./" == target_filename[0:2]:
                target_filename = target_filename[2:]
            target_img = cv2.imread(self.path_meta+ "/images/" + target_filename)
            prompt = item['prompt']
            self.none_loop += 1
            if self.none_loop > 10000:
                break
                
        source_img,  sizes = self.resize_image_control(source_img, self.resolution)
        target_img = self.resize_image_target(target_img, self.resolution, sizes)
        
        # Do not forget that OpenCV read images in BGR order.
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source_img = source_img.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target_img = (target_img.astype(np.float32) / 127.5) - 1.0
        
        prompt = prompt if random.uniform(0, 1) > 0.3 else '' # dropout rate 30%
        return dict(jpg=target_img, txt=prompt, hint=source_img, task=self.key_prompt)

