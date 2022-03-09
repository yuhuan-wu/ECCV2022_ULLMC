# Segmentation Decoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
import math
import cv2
import numpy as np
import os
# from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmcv.cnn import normal_init
from mmcv.runner import auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from mmseg.models.builder import build_loss
from mmseg.models.losses import accuracy

from abc import ABCMeta, abstractmethod

from mmseg.models.utils.layers import WindowAttention, WindowCrossAttention
import torch.utils.checkpoint as checkpoint
from einops import rearrange

@HEADS.register_module()
class GTR_DECODER3(nn.Module):
    """
    dim_feedforwards: [256,512,1024]
    d_models: [64,128,256]
    patch_sizes: [[2,2]... [8,8]]
    """

    def __init__(self, embed_dim=96, query_dim=96, memory_dims=[96, 192, 384],
                 depths=[1, 1, 1], num_heads=[8, 8, 8], window_size_q=(7, 7), window_size_k=(7, 7),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 num_classes=20,
                 aux_classes=2,
                 align_corners=False,
                 head_scale=1,
                 use_checkpoint=False, **kwargs):
        super(GTR_DECODER3, self).__init__(**kwargs)
        # to do
