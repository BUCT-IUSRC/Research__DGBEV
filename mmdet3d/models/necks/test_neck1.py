import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.builder import NECKS

__all__ = ["Test_Neck_1"]


@NECKS.register_module()
class Test_Neck_1(nn.Module):
    pass