'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
'''

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from dataset_eval import MyDataset
from cldm.model import create_model, load_state_dict
from pathlib import Path
import jsonlines
import argparse
import pdb
from PIL import Image
import numpy as np
import einops
import os
from cldm.ddim_unicontrol_hacked import DDIMSampler
import random
from torchvision.utils import make_grid
from utils import check_safety

parser = argparse.ArgumentParser(description="args")
parser.add_argument("--task", type=str, default='canny', choices=['canny', 'hed', 'seg', 'normal', 'depth','openpose', 'imageedit', 'bbox', 'hedsketch', 'outpainting', 'grayscale', 'blur', 'inpainting', 'grayscale'], help='option of task')
parser.add_argument("--ckpt", type=str, default='./ckpts/unicontrol.ckpt', help='$path to checkpoint')
parser.add_argument("--strength", type=float, default=1.0, help='control guidiance strength')
parser.add_argument("--scale", type=float, default=9.0, help='text guidiance scale')
parser.add_argument("--output_path", type=str, default='./output', help='$path to save prediction results')
parser.add_argument("--config", type=str, default='./models/cldm_v15_unicontrol.yaml', help='option of config') 
parser.add_argument("--guess_mode", default=False, help='Guess Mode') 
parser.add_argument("--seed", default=-1, help='Random Seed') 
parser.add_argument("--save_memory", default=False, help='Low Memory') 
parser.add_argument("--num_samples", type=int, default=3, help='Num of Samples') 
parser.add_argument("--n_prompt", type=str, default='worst quality, low quality', help='negative prompts') 
parser.add_argument("--ddim_steps", default=50, help='DDIM Steps') 

args = parser.parse_args()

# Configs
checkpoint_path = args.ckpt
batch_size = 1
seed = args.seed
num_samples = args.num_samples
guess_mode = args.guess_mode
n_prompt = args.n_prompt
ddim_steps=args.ddim_steps

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(args.config).cpu()
model.load_state_dict(load_state_dict(checkpoint_path, location='cpu'), strict=False) #, strict=False

task=args.task


output_dir = os.path.join(args.output_path, 'scale'+str(int(args.scale)), task)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
control_key = 'control_' + task

path_meta= "data/"
# task_name = task if task != 'seg' else 'segbase'
task_name = task
path_json = "data/" + task_name + ".json"

target_list = []
with jsonlines.open(Path( path_json)) as reader:
    for ll in reader:
        target_list.append(ll[control_key].split('/')[1])
        
print(f"Length of target list is {len(target_list)}")

model.eval()

dataset = MyDataset(path_json, path_meta, task_name )

dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
ddim_sampler = DDIMSampler(model)
    
sample_path = os.path.join(output_dir, "samples")
os.makedirs(sample_path, exist_ok=True)
base_count = len(os.listdir(sample_path))
grid_count = len(os.listdir(output_dir)) - 1

a_prompt = 'best quality, extremely detailed'
# Inference loop
with torch.no_grad():
    for idx, batch in enumerate(dataloader):
        prompt = batch['txt'][0]
        
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if args.save_memory:
            model.low_vram_shift(is_diffusing=False)
        
        control = batch['hint'].squeeze(0).cuda() # torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        H, W, C = control.shape
        
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        
        task_dic = {}
        task_dic['name'] = batch['task'][0]
        task_instruction = batch['instruction'][0]
        task_dic['feature'] = model.get_learned_conditioning(task_instruction)[:,:1,:]
        
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], "task": task_dic}
        un_cond = {"c_concat": [torch.zeros_like(control)] if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([""] * num_samples)]}
        shape = (4, H // 8, W // 8)
        
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=0,
                                                     unconditional_guidance_scale=args.scale,
                                                     unconditional_conditioning=un_cond)
        x_samples = model.decode_first_stage(samples)

        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
        
        x_checked_image, has_nsfw_concept = check_safety(x_samples)
        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
        for x_sample in x_checked_image_torch:
            x_sample = 255. * einops.rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            img.save(os.path.join(sample_path, prompt.replace(" ", "-")[:-1] +'-' + f"{base_count:05}" + ".png"))
            base_count += 1
        control_img = Image.fromarray((batch['hint'].squeeze(0).cpu().numpy() *  255.0).astype(np.uint8))
        control_img.save(os.path.join(sample_path, prompt.replace(" ", "-")[:-1] + '-'+ 'control' + ".png"))