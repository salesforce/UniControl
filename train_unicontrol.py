'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
'''

from share import *
from torch.utils.data.dataset import ConcatDataset

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from train_util.dataset import MyDataset
from cldm.logger import ImageLogger, CheckpointEveryNSteps
from cldm.model import create_model, load_state_dict
from train_util.multi_task_scheduler import BatchSchedulerSampler
import train_util.dataset_collate as dataset_collate
import os

import argparse

parser = argparse.ArgumentParser(description="args")
parser.add_argument("--ckpt", type=str, default='./ckpts/control_sd15_ini.ckpt', help='path to checkpoint')
parser.add_argument("--config", type=str, default='./models/cldm_v15_unicontrol_v11.yaml', help='config')
parser.add_argument("--lr", type=float, default=1e-5, help='learning rate')
parser.add_argument("--bs", type=int, default=4, help='batchsize')
parser.add_argument("--img_logger_freq", type=int, default=1000, help='img logger freq')
parser.add_argument("--ckpt_logger_freq", type=int, default=20000, help='ckpt logger freq')
parser.add_argument("--data_path", type=str, default='./multigen20m/dataset', help='path to dataset')

args = parser.parse_args()

# Configs
resume_path = args.ckpt #'../checkpoints_v1/control_sd15_ini.ckpt'
config = args.config
ckpt_logger_freq = args.ckpt_logger_freq
batch_size = args.bs
logger_freq = args.img_logger_freq
learning_rate = args.lr
sd_locked = True
only_mid_control = False

# Construct Model
model = create_model(config).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Construct Training Datasets
tasks=['hed', 'canny', 'depth'] 
# tasks_all=['hed', 'canny', 'hedsketch', 'depth', 'normal', 'seg', 'bbox', 'openpose', 'outpainting', 'inpainting', 'blur', 'grayscale']
datasets_list = []
path_meta= args.data_path 
for _task in tasks:
    path_json = os.path.join(args.data_path, 'json_files',  'aesthetics_plus_all_group_'+_task +'_all.json')
    datasets_list.append(MyDataset(path_json, path_meta, _task))
    
multi_dataset = ConcatDataset(datasets_list)
dataloader = DataLoader(multi_dataset, num_workers=16,  sampler=BatchSchedulerSampler(dataset=multi_dataset, batch_size=batch_size), batch_size=batch_size, persistent_workers=True, shuffle=False, collate_fn=dataset_collate.collate_fn)

# Construct Training Logger
logger_img = ImageLogger(batch_frequency=logger_freq)
logger_checkpoint = CheckpointEveryNSteps(save_step_frequency=ckpt_logger_freq)

# Build Trainer
trainer = pl.Trainer(gpus=-1, precision=16, accelerator='ddp', callbacks=[logger_img,logger_checkpoint], replace_sampler_ddp=False)

if __name__ == '__main__':    
    # Train!
    trainer.fit(model, dataloader)
