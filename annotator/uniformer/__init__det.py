# Uniformer
# From https://github.com/Sense-X/UniFormer
# # Apache-2.0 license

import os

# from annotator.uniformer.mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
# from annotator.uniformer.mmseg.core.evaluation import get_palette
# from annotator.util import annotator_ckpts_path

from annotator.uniformer.mmdet.apis import init_detector, inference_detector, show_result_pyplot
from annotator.uniformer.mmdet.core.evaluation import get_palette
from annotator.util import annotator_ckpts_path
    
# checkpoint_file = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/upernet_global_small.pth"


class UniformerDetector:
    def __init__(self):
        modelpath = os.path.join(annotator_ckpts_path, "cascade_mask_rcnn_3x_ms_hybrid_base.pth")

        config_file = os.path.join(os.path.dirname(annotator_ckpts_path), "uniformer", "exp", "cascade_mask_rcnn_3x_ms_hybrid_base", "config.py")
        self.model = init_detector(config_file, modelpath).cuda()

    def __call__(self, img):
        result = inference_detector(self.model, img)
        res_img = show_result_pyplot(self.model, img, result, get_palette('coco'), opacity=1)
        return res_img
