# [UniControl](https://arxiv.org/abs/2305.11147) [![arXiv](https://img.shields.io/badge/📃-arXiv-ff69b4)](https://arxiv.org/pdf/2305.11147.pdf) [![webpage](https://img.shields.io/badge/🖥-Website-9cf)](https://canqin001.github.io/UniControl-Page/)
<div align="center">
    <a><img src="figs/salesforce.png"  height="100px" ></a>
    <a><img src="figs/northeastern.png"  height="100px" ></a>
    <a><img src="figs/stanford.png"  height="100px" ></a>
</div>

This repository is for the paper:
> **[UniControl: A Unified Diffusion Model for Controllable Visual Generation In the Wild](https://arxiv.org/abs/2305.11147)** \
> Can Qin <sup>1,2</sup>, Shu Zhang<sup>1</sup>, Ning Yu <sup>1</sup>, Yihao Feng<sup>1</sup>, Xinyi Yang<sup>1</sup>, Yingbo Zhou <sup>1</sup>, Huan Wang <sup>1</sup>, Juan Carlos Niebles<sup>1</sup>, Caiming Xiong <sup>1</sup>, Silvio Savarese <sup>1</sup>, Stefano Ermon <sup>3</sup>, Yun Fu <sup>2</sup>,  Ran Xu <sup>1</sup> \
> <sup>1</sup> Salesforce AI <sup>2</sup> Northeastern University  <sup>3</sup> Stanford Univerisy \
> Work done when Can Qin was an intern at Salesforce AI Research.


## Instruction
### Environment Preparation
Setup the env first (need to wait a few minutes).
```
conda env create -f environment.yaml
conda activate unicontrol
```
### Checkpoint Preparation (Only For Training)
Then you need to decide which Stable Diffusion Model you want to control. In this example, we will just use standard SD1.5. You can download it from the [official page of Stability](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main). You want the file ["v1-5-pruned.ckpt"](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main).

(Or ["v2-1_512-ema-pruned.ckpt"](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main) if you are using SD2.)

Note that all weights inside the ControlNet are also copied from SD so that no layer is trained from scratch, and you are still finetuning the entire model.

We provide a simple script for you to achieve this easily. If your SD filename is "./models/v1-5-pruned.ckpt" and you want the script to save the processed model (SD+ControlNet) at location "./models/control_sd15_ini.ckpt", you can just run:

    python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt

Or if you are using SD2:

    python tool_add_control_sd21.py ./models/v2-1_512-ema-pruned.ckpt ./models/control_sd21_ini.ckpt
 
The checkpoint of pre-trained model is saved at "laion400m-data/canqin/checkpoints_v1/ours_latest_acti.ckpt".
    
### Data Preparation 
need volume "laion400m-data-ssd" for tasks "canny, hed, seg, depth, normal, depth, openpose".

### Model Inference (CUDA 11.0 and Conda 4.12.0 work)
For different tasks, please run the code as follows. If you meet OOM error, please decrease the "--num_samples".

Canny to Image Generation:
```
python inference_demo.py --ckpt ../checkpoints_v1/ours_latest_acti.ckpt --task canny 

```

HED Edge to Image Generation:
```
CUDA_VISIBLE_DEVICES=2 python inference_demo.py --ckpt ../checkpoints_v1/ours_latest_acti.ckpt --task hed 
```

HED-like Skech to Image Generation:
```
CUDA_VISIBLE_DEVICES=0 python inference_demo.py --ckpt ../checkpoints_v1/ours_latest_acti.ckpt --task hedsketch
```


Depth Map to Image Generation:
```
CUDA_VISIBLE_DEVICES=3 python inference_demo.py --ckpt ../checkpoints_v1/ours_latest_acti.ckpt --task depth 
```

Normal Surface Map to Image Generation:
```
CUDA_VISIBLE_DEVICES=4 python inference_demo.py --ckpt ../checkpoints_v1/ours_latest_acti.ckpt --task normal
```

Segmentation Map to Image Generation:
```
python inference_demo.py --ckpt ../checkpoints_v1/ours_latest_acti.ckpt --task seg
```

Human Skeleton to Image Generation:
```
CUDA_VISIBLE_DEVICES=1 python inference_demo.py --ckpt ../checkpoints_v1/ours_latest_acti.ckpt --task openpose
```

Object Bounding Boxes to Image Generation:
```
CUDA_VISIBLE_DEVICES=2 python inference_demo.py --ckpt ../checkpoints_v1/ours_latest_acti.ckpt --task bbox
```

Image Outpainting:
```
CUDA_VISIBLE_DEVICES=3 python inference_demo.py --ckpt ../checkpoints_v1/ours_latest_acti.ckpt --task outpainting
```


### Model Training (CUDA 11.0 and Conda 4.12.0 work)
For single task, please run the following code with your options of "task" and it will use GPU (DDP):
```
python train_single_task.py --task canny --checkpoint_path ./models/control_sd15_ini.ckpt
```
then the model checkpoint will be saved at "lightning_logs/version_$num" and image logger visualization will apprear in "image_log/train".

For multi task, please run the following code with your options of "task" and it will use GPU (DDP):
```
python train_multi_task_full.py
```
then the model checkpoint will be saved at "lightning_logs/version_$num" and image logger visualization will apprear in "image_log/train".

## To Do
- [x] Data Preparation
- [x] Pre-training Tasks Inference
    - [x] Canny-to-image
    - [x] HED-to-image
    - [x] HEDSketch-to-image
    - [x] Depth-to-image
    - [x] Normal-to-image
    - [x] Seg-to-image
    - [x] Human-Skeleton-to-image
    - [x] Bbox-to-image
    - [x] Image-outpainting
- [ ] Model Training
- [ ] Zero-shot Tasks Inference
- [ ] Jupyter Notebook



    