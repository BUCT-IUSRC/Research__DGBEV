import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.builder import NECKS

__all__ = ["cam_feature_without_depth"]


@NECKS.register_module()
class cam_feature_without_depth(nn.Module):
    def __init__(self):
        # super(self).__init__()
        super().__init__()
        self.conv = nn.Conv2d(768, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.interpolate(x, size=(180, 180), mode='bilinear', align_corners=False)
        return x