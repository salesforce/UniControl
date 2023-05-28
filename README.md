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


![img](figs/demo_simple.png)



## Introduction
Achieving machine autonomy and human control often represent divergent objectives in the design of interactive AI systems. Visual generative foundation models such as Stable Diffusion show promise in navigating these goals, especially when prompted with arbitrary languages. However, they often fall short in generating images with spatial, structural, or geometric controls. The integration of such controls, which can accommodate various visual conditions in a single unified model, remains an unaddressed challenge. In response, we introduce UniControl, a new generative foundation model that consolidates a wide array of controllable condition-to-image (C2I) tasks within a singular framework, while still allowing for arbitrary language prompts. UniControl enables pixel-level-precise image generation, where visual conditions primarily influence the generated structures and language prompts guide the style and context. To equip UniControl with the capacity to handle diverse visual conditions, we augment pretrained text-to-image diffusion models and introduce a task-aware HyperNet to modulate the diffusion models, enabling the adaptation to different C2I tasks simultaneously. Trained on nine unique C2I tasks, UniControl demonstrates impressive zero-shot generation abilities with unseen visual conditions. Experimental results show that UniControl often surpasses the performance of single-task-controlled methods of comparable model sizes. This control versatility positions UniControl as a significant advancement in the realm of controllable visual generation. 


![img](figs/method.png)

## Updates
* 05/18/23: [UniControl](https://arxiv.org/abs/2305.11147) paper uploaded to arXiv.
* 05/26/23: UniControl inference code and checkpoint is open to public.
* 05/28/23: New UniControl checkpoint is updated.


## Instruction
### Environment Preparation
Setup the env first (need to wait a few minutes).
```
conda env create -f environment.yaml
conda activate unicontrol
```
### Checkpoint Preparation
The checkpoint of pre-trained UniControl model is saved at `./ckpts/unicontrol.ckpt`.
```
mkdir ckpts
cd ckpts
wget https://storage.googleapis.com/sfr-unicontrol-data-research/unicontrol.ckpt 
```

<!-- 
If you want to train from scratch, please follow the ControlNet to prepare the checkpoint initialization.

ControlNet provides a simple script for you to achieve this easily. If your SD filename is `./models/v1-5-pruned.ckpt` and you want the script to save the processed model (SD+ControlNet) at location `./models/control_sd15_ini.ckpt`, you can just run:

    python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt

Or if you are using SD2:

    python tool_add_control_sd21.py ./models/v2-1_512-ema-pruned.ckpt ./models/control_sd21_ini.ckpt -->
 
### Data Preparation 
The example inference data already are saved at `./data` and `./test_imgs_CN`.

### Model Inference (CUDA 11.0 and Conda 4.12.0 work)
For different tasks, please run the code as follows. If you meet OOM error, please decrease the "--num_samples".

Canny to Image Generation:
```
python inference_demo.py --ckpt ./ckpts/unicontrol.ckpt --task canny 
```

HED Edge to Image Generation:
```
python inference_demo.py --ckpt ./ckpts/unicontrol.ckpt --task hed 
```

HED-like Skech to Image Generation:
```
python inference_demo.py --ckpt ./ckpts/unicontrol.ckpt --task hedsketch
```

Depth Map to Image Generation:
```
python inference_demo.py --ckpt ./ckpts/unicontrol.ckpt --task depth 
```

Normal Surface Map to Image Generation:
```
python inference_demo.py --ckpt ./ckpts/unicontrol.ckpt --task normal
```

Segmentation Map to Image Generation:
```
python inference_demo.py --ckpt ./ckpts/unicontrol.ckpt --task seg
```


Human Skeleton to Image Generation:
```
python inference_demo.py --ckpt ./ckpts/unicontrol.ckpt --task openpose
```


Object Bounding Boxes to Image Generation:
```
python inference_demo.py --ckpt ./ckpts/unicontrol.ckpt --task bbox
```


Image Outpainting:
```
python inference_demo.py --ckpt ./ckpts/unicontrol.ckpt --task outpainting
```

### Gradio Demo ([App Demo Video](https://github.com/salesforce/UniControl/issues/1), CUDA 11.0 and Conda 4.12.0 work)
We have provided gradio demos for different tasks to use. The example images are saved at `./test_imgs`. 
<div align="center">
    <a><img src="figs/gradio_all_tasks.png"  height="300px" ></a>
</div>

For all the tasks (`Canny, HED, Sketch, Depth, Normal, Human Pose, Seg, Bbox, Outpainting`) please run the following code: 
```
python gradio_all_tasks.py
```

<div align="center">
    <a><img src="figs/gradio_all_task_sketch.png"  height="300px" ></a>
</div>

We support the direct condition-to-image generation (as shown above). Please unmark the `Condition Extraction` in UI if you want to upload condition image directly. 


Or, we provide the task-specifc gradio demos:

<div align="center">
    <a><img src="figs/gradio_canny.png"  height="300px" ></a>
</div>

Canny to Image Generation:
```
python gradio_canny2image.py
```

<div align="center">
    <a><img src="figs/gradio_hed.png"  height="300px" ></a>
</div>

HED Edge to Image Generation:
```
python gradio_hed2image.py
```

<div align="center">
    <a><img src="figs/gradio_hedsketch.png"  height="300px" ></a>
</div>

HED-like Skech to Image Generation:
```
python gradio_hedsketch2image.py
```

<div align="center">
    <a><img src="figs/gradio_depth.png"  height="300px" ></a>
</div>

Depth Map to Image Generation:
```
python gradio_depth2image.py
```

<div align="center">
    <a><img src="figs/gradio_normal.png"  height="300px" ></a>
</div>

Normal Surface Map to Image Generation:
```
python gradio_normal2image.py
```

<div align="center">
    <a><img src="figs/gradio_seg.png"  height="300px" ></a>
</div>

Segmentation Map to Image Generation:
```
python gradio_seg2image.py
```

<div align="center">
    <a><img src="figs/gradio_pose.png"  height="300px" ></a>
</div>

Human Skeleton to Image Generation:
```
python gradio_pose2image.py
```

<div align="center">
    <a><img src="figs/gradio_bbox.png"  height="300px" ></a>
</div>

Object Bounding Boxes to Image Generation:
```
python gradio_bbox2image.py
```

<div align="center">
    <a><img src="figs/gradio_outpainting.png"  height="300px" ></a>
</div>

Image Outpainting:
```
python gradio_outpainting.py
```


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
- [x] Gradio Demo
- [ ] Zero-shot Tasks Inference
- [ ] Model Training

## Tips
* Negative prompts are very useful in some cases: `monochrome, lowres, bad anatomy, worst quality, low quality` are example ones.


## Citation
If you find this project useful for your research, please kindly cite our paper:

```bibtex
@article{qin2023unicontrol,
  title={UniControl: A Unified Diffusion Model for Controllable Visual Generation In the Wild},
  author={Qin, Can and Zhang, Shu and Yu, Ning and Feng, Yihao and Yang, Xinyi and Zhou, Yingbo and Wang, Huan and Niebles, Juan Carlos and Xiong, Caiming and Savarese, Silvio and others},
  journal={arXiv preprint arXiv:2305.11147},
  year={2023}
}
```

## Acknowledgement
*This project is built upon the gaint sholders of ControlNet and Stable Diffusion. Great thanks to them.*

Stable Diffusion https://github.com/CompVis/stable-diffusion

ControlNet https://github.com/lllyasviel/ControlNet 

StyleGAN3 https://github.com/NVlabs/stylegan3
    