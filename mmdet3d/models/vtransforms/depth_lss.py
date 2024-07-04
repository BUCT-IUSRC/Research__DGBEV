from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseDepthTransform
import json
import torch.nn.functional as F
import numpy as np
__all__ = ["DepthLSSTransform"]



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        max_out = self.max_pool(x).view(x.size(0), -1)
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        out = avg_out + max_out
        return out

class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super(AttentionModule,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.channel_attention = ChannelAttention(out_channels, reduction_ratio)

    def forward(self, x):
        x = self.conv(x)
        x_attention= self.channel_attention(x)
        x_attention = x_attention.view(x_attention.size(0), x_attention.size(1), 1, 1)  # 扩展维度
        x = x * x_attention
        return x
def fuse_d_depth(depth1,depths_org):
        # 找到 d 中的零值的索引
        depth1 = depth1.cpu().numpy()
        depths_org = depths_org.cpu().numpy()
        zero_indices = np.where(depth1 == 0)
        # d_copy= np.copy(d)
        # 使用 depths_org 中对应位置的值来填充 d
        depth1[zero_indices] = depths_org[zero_indices]        
        depth1 = torch.from_numpy(depth1).float()
        pool_size=2
        processed_fill_d_tensor = F.avg_pool2d(depth1, pool_size, stride=pool_size)
        upsampled_tensor = F.interpolate(processed_fill_d_tensor, scale_factor=pool_size, mode='nearest')
        return upsampled_tensor
@VTRANSFORMS.register_module()
class DepthLSSTransform(BaseDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential(
            nn.Conv2d(2, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depth_map_transform = nn.Sequential(

            nn.Conv2d(1, 1, 5, stride=4, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Conv2d(1 ,1, 5, stride=2, padding=2),

           
        )
        
        self.prenet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            
        )
        # self.conv_depthmap = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 5), stride=(3, 18), padding=(1, 0))
        self.depthnet=nn.Conv2d(in_channels, self.D , 1)
        self.contextnet = nn.Conv2d(in_channels, self.C, 1)
        self.attention1 = AttentionModule(in_channels, 118)
        # self.attention1 = nn.Conv2d(in_channels, 1, 1)
        # self.attention2 = nn.Conv2d(self.C, 1, 1)

        self.convnet = nn.Sequential(  # 提取图像语义
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.C, 3, padding=1),
            nn.BatchNorm2d(self.C),
            nn.ReLU(True),
        )
        
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()

    
    
    def get_cam_feats(self, x, d,depths_org):
        B, N, C, fH, fW = x.shape
        # print("!!!!!!!!!!!!!depthlss!!!!!!!!!!!!!!",depths_org.shape,d.shape)

        # print(len(depths_org))#6
        # print("DepthLSSTransform_get_cam_feats x_input shape :")
        # print(x.size())torch.Size([2, 6, 256, 32, 88])
        #print(type(x))
        # print("DepthLSSTransform_get_cam_feats d_input shape :")
        # print(d.size())torch.Size([2, 6, 1, 256, 704])
        #print(type(d))
        d = d.view(B * N, *d.shape[2:])#torch.Size([12, 64, 32, 88])
        x = x.view(B * N, C, fH, fW)
        depths_org=depths_org.view(B * N, 1, 256, 704)/257
        fuse_d=fuse_d_depth(d,depths_org).cuda()
        fuse_d_upsample=self.depth_map_transform(fuse_d)
        # print(fuse_d_upsample.shape)torch.Size([12, 1, 32, 88])
        # igig
        d=torch.cat([d, fuse_d], dim=1)
        # i=1
        # while i<2:
            
        #     i=2
        #     # 将张量转换为Python列表
        #     d_list = d[:2,:,:,:].tolist()
        #     depths_org_list = depths_org[:2,:,:,:].tolist()

        #     # 将两个列表存储到字典中
        #     data = {"d_list": d_list, "depths_org_list": depths_org_list}

        #     # 将字典保存为JSON文件
        #     with open("/data/workdirs/bevfusion/depth_map2/d-depth.json", "w") as json_file:
        #         json.dump(data, json_file)
        #     print("#########save##########")
        #     vdvd
       # depths_org=self.depth_map_transform(depths_org)
        #print('d.view :')
        #print(d.shape)
        #print('x.view :')
        #print(x.shape)
        d = self.dtransform(d)
        context=self.convnet(x)# torch.Size([12, 80, 32, 88])
        x = torch.cat([d, x], dim=1)# torch.Size([12, 320, 32, 88])
        
        
        x = self.prenet(x)#torch.Size([12,198, 32, 88])
        x_attention =self.attention1(x)#torch.Size([12, 64, 32, 88])
        
        # [12, 118, 32, 88])
        
        depth=self.depthnet(x)
        depth=depth*x_attention
        depth=(depth*fuse_d_upsample+depth).softmax(dim=1)#([12, 64, 32, 88])
     
        x = self.contextnet(x) + context#torch.Size([12, 80, 32, 88])
       
        
        # attention2 = self.attention2(x)#([12, 1, 32, 88])    
        x = depth.unsqueeze(1) * (x).unsqueeze(2)#[12, 80, 64, 32, 88])
        # print("inchannals  c d")
        # print(self.in_channels,self.C,self.D)#256 80 118
       
        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)

        # print("DepthLSSTransform_get_cam_feats output shape :")
        # print(x.size())
        return x

    def forward(self, *args, **kwargs):
        # print("DepthLSSTransform details")
        # print(self.in_channels, self.out_channels, self.image_size, self.feature_size, self.xbound, self.ybound, self.zbound, self.dbound)

        x = super().forward(*args, **kwargs)
        # print(x.shape)
        x = self.downsample(x)
        # print("$$$$$$$$$$$$$$$$$$$$")
        # print(x.shape)
        # print("DepthLSSTransform output shape :")
        # print(x.size())
        return x