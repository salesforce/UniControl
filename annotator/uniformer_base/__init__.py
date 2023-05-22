'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
 * Modified from ControlNet repo: https://github.com/lllyasviel/ControlNet
 * Copyright (c) 2023 Lvmin Zhang and Maneesh Agrawala
 * Modified from UniFormer repo: From https://github.com/Sense-X/UniFormer
 * Apache-2.0 license
'''



import os
from annotator.uniformer.mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from annotator.uniformer.mmseg.core.evaluation import get_palette
from annotator.util import annotator_ckpts_path

import pdb
# checkpoint_file = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/upernet_global_small.pth"

class UniformerDetector:
    def __init__(self):
        modelpath = os.path.join(annotator_ckpts_path, "upernet_global_base.pth")
        if not os.path.exists(modelpath):
#             from basicsr.utils.download_util import load_file_from_url
#             load_file_from_url(checkpoint_file, model_dir=annotator_ckpts_path)
            raise ValueError("wrong ckpt path")
            
        config_file = os.path.join(os.path.dirname(annotator_ckpts_path), "uniformer_base", "exp", "upernet_global_base", "config.py")
        self.model = init_segmentor(config_file, modelpath).cuda()

    def __call__(self, img):
        result = inference_segmentor(self.model, img)
        res_img = show_result_pyplot(self.model, img, result, get_palette('ade'), opacity=1)
        return res_img
