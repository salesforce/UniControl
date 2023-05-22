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
    def __init__(self, path_json, path_meta, task ): # './training/fill50k/prompt.json'
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
        elif task == 'bbox':
            self.key_prompt = 'control_bbox'
        elif task == 'grayscale':
            self.key_prompt = 'control_grayscale'
        elif task == 'outpainting':
            self.key_prompt = 'control_outpainting'
        elif task == 'hedsketch':
            self.key_prompt = 'control_hedsketch'
        self.key_prompt_model = self.key_prompt   
        
        self.task_to_instruction = {"control_hed": "hed edge to image", "control_canny": "canny edge to image", "control_seg": "segmentation map to image", "control_depth": "depth map to image", "control_normal": "normal surface map to image", "control_img": "image editing", "control_openpose": "human pose skeleton to image", "control_hedsketch": "sketch to image", "control_bbox": "bounding box to image", "control_outpainting": "image outpainting"}

        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_filename = item[self.key_prompt]
        print(self.path_meta + source_filename)
        source_img = cv2.imread(self.path_meta + source_filename)
        target_filename = item['source']
        if "./" == target_filename[0:2]:
            target_filename = target_filename[2:]
            
#         target_img = cv2.imread(self.path_meta + target_filename)
        prompt = item['prompt']

        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
#         target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source_img = source_img.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
#         target_img = (target_img.astype(np.float32) / 127.5) - 1.0

        return dict(txt=prompt, hint=source_img, task=self.key_prompt_model, instruction=self.task_to_instruction[self.key_prompt_model])

