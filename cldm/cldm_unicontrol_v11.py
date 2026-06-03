'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Can Qin
 * Modified from ControlNet repo: https://github.com/lllyasviel/ControlNet
 * Copyright (c) 2023 Lvmin Zhang and Maneesh Agrawala
'''

import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm_multi import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim_multi import DDIMSampler

import pdb
import copy

def modulated_conv2d(
    x,                  # Input tensor: [batch_size, in_channels, in_height, in_width]
    w,                  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
    s,                  # Style tensor: [batch_size, in_channels]
    demodulate  = False, # Apply weight demodulation?
    padding     = 0,    # Padding: int or [padH, padW]
    input_gain  = None, # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
    bias=None, 
    stride=1, 
    dilation=1
):
    """
    https://github.com/NVlabs/stylegan3/blob/407db86e6fe432540a22515310188288687858fa/training/networks_stylegan3.py
    """
#     with misc.suppress_tracer_warnings(): # this value will be treated as a constant
    batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape

    # Modulate weights.
    w = w.unsqueeze(0) # [NOIkk]
    w = (w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4)) # [NOIkk]
    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    
    x = torch.nn.functional.conv2d(input=x, weight=w.to(x.dtype), bias=bias, stride=stride, padding=padding, dilation=dilation, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


    
class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            all_tasks_num = 13,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'
        
        self.all_tasks_num = all_tasks_num
        self.tasks_to_id = {"control_hed":0, "control_canny":1, "control_seg":2, "control_depth":3, "control_normal":4,"control_openpose":5, "control_img":6, "control_hedsketch":7, "control_bbox":8, "control_outpainting":9,  "control_grayscale":10,  "control_blur":11, "control_inpainting":12}
        
        
        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        self.task_id_hypernet = nn.Sequential(
            linear(768, time_embed_dim), # model_channels or 768
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        self.task_id_layernet = []
        

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.task_id_layernet.append(linear(time_embed_dim, model_channels))
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)]) # ie, model_channels -> 320
        
        self.input_hint_block_list_moe = nn.ModuleList([TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU()
        ) for _ in range( self.all_tasks_num)])
        
        self.input_hint_block_zeroconv_0 = nn.ModuleList([(conv_nd(dims, 32, 32, 3, padding=1)),(conv_nd(dims, 32, 32, 3, padding=1))]) # non-zero conv is helful in the early stage training
        
        self.task_id_layernet_zeroconv_0 = linear(time_embed_dim, 32)
        self.input_hint_block_share = TimestepEmbedSequential(
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
        ) 
        
        self.input_hint_block_zeroconv_1 = nn.ModuleList([(conv_nd(dims, 256, model_channels, 3, padding=1)), (conv_nd(dims, 256, model_channels, 3, padding=1)) ])  # non-zero conv is helful in the early stage training
        
        self.task_id_layernet_zeroconv_1 = linear(time_embed_dim, 256)
        
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                
                self.task_id_layernet.append(linear(time_embed_dim, ch))
                self.zero_convs.append(self.make_zero_conv(ch))
                
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                
                self.task_id_layernet.append(linear(time_embed_dim, ch))
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch
                

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch
        self.task_id_layernet = nn.ModuleList(self.task_id_layernet)
    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        
        '''
        x -> 4,4,64,64
        hint -> 4, 3, 512, 512
        context - > 4, 77, 768
        '''
        BS = 1 # x.shape[0], one batch one task
        BS_Real = x.shape[0]
        if kwargs is not None:
            task_name = kwargs['task']['name']
            task_id = self.tasks_to_id[task_name]
            task_feature = kwargs['task']['feature']
            task_id_emb = self.task_id_hypernet(task_feature.squeeze(0))
            
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        guided_hint = self.input_hint_block_list_moe[task_id](hint, emb, context)
        
        guided_hint = modulated_conv2d(guided_hint, self.input_hint_block_zeroconv_0[0].weight, self.task_id_layernet_zeroconv_0(task_id_emb).repeat(BS_Real, 1), padding=1) + self.input_hint_block_zeroconv_0[1](guided_hint)
        # self.input_hint_block_zeroconv_0[1] is helful in the early stage training
        
        guided_hint += self.input_hint_block_zeroconv_0[0].bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        guided_hint = self.input_hint_block_share(guided_hint, emb, context)
        guided_hint = modulated_conv2d(guided_hint, self.input_hint_block_zeroconv_1[0].weight, self.task_id_layernet_zeroconv_1(task_id_emb).repeat(BS_Real, 1), padding=1) + self.input_hint_block_zeroconv_1[1](guided_hint)
        # self.input_hint_block_zeroconv_1[1] is helful in the early stage training
        
        guided_hint += self.input_hint_block_zeroconv_1[0].bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        outs = []
        h = x.type(self.dtype)
        for module, zero_conv, task_hyperlayer in zip(self.input_blocks, self.zero_convs, self.task_id_layernet):
            if guided_hint is not None:
                h = module(h, emb, context)
                try:
                    h += guided_hint
                except RuntimeError:
                    pdb.set_trace()
                guided_hint = None
            else:
                h = module(h, emb, context)            
            outs.append(modulated_conv2d(h, zero_conv[0].weight, task_hyperlayer(task_id_emb).repeat(BS_Real, 1)) + zero_conv[0].bias.unsqueeze(0).unsqueeze(2).unsqueeze(3))
        
        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapping_task = {"control_hed": "hed edge to image", "control_canny": "canny edge to image", "control_seg": "segmentation map to image", "control_depth": "depth map to image", "control_normal": "normal surface map to image", "control_img": "image editing", "control_openpose": "human pose skeleton to image", "control_hedsketch": "sketch to image", "control_bbox": "bounding box to image", "control_outpainting": "image outpainting", "control_grayscale": "gray image to color image", "control_blur": "deblur image to clean image", "control_inpainting": "image inpainting"}
        self.all_tasks_num = len(self.mapping_task)
#         self.task_weight_all = nn.Parameter(torch.zeros(self.all_tasks_num,), requires_grad=True)
        self.task_loss_ema = torch.zeros(self.all_tasks_num,)
        
        self.control_model = instantiate_from_config(control_stage_config) # -> ControlNet
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        
        '''
        self -> ControlLDM(
        (model): DiffusionWrapper(
        (diffusion_model): ControlledUnetModel(...)
        (first_stage_model): AutoencoderKL(...)
        (cond_stage_model): FrozenCLIPEmbedder(...)
        (control_model): ControlNet(...)
        batch - > dict('jpg', 'txt', 'hint', 'task')
        '''

        task_name = batch['task'][0] # one task for one batch
        BS = len(batch['task'])
        batch['txt'] = batch['txt'] + [self.mapping_task[task_name]]
        
        x, c_all = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        c, c_task = c_all[:BS,:,:], c_all[BS:BS+1,:1,:]
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        task_dic = {}
        task_dic['name'] = task_name
        task_dic['feature'] = c_task
        return x, dict(c_crossattn=[c], c_concat=[control], task=task_dic)

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        task_name = cond['task'] # dict['name', 'feature']
        diffusion_model = self.model.diffusion_model # -> ControlledUnetModel

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        
        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt, task=task_name)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()

        task_name = batch['task'][0] # one task for one batch
        
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
                
        task_dic = c['task']
        
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], 'task': task_dic},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log
    
    @torch.no_grad()
    def log_images_infer(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()

        task_name = batch['task'][0] # one task for one batch
        
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
                
        task_dic = c['task']
        
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
#         log["reconstruction"] = self.decode_first_stage(z)
#         log["control"] = c_cat * 2.0 - 1.0
#         log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        

        uc_cross = self.get_unconditional_conditioning(N)
        uc_cat = c_cat  # torch.zeros_like(c_cat)
        uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
        samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], 'task': task_dic},
                                         batch_size=N, ddim=use_ddim,
                                         ddim_steps=ddim_steps, eta=ddim_eta,
                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                         unconditional_conditioning=uc_full,
                                         )
        x_samples_cfg = self.decode_first_stage(samples_cfg)
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
