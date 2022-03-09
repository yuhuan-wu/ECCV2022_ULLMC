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
