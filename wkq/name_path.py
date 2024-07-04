import os

import numpy as np
import torch
# seg_path = os.path.join(
#         *tokens[:-3],
#         "virtual_points",
#         tokens[-3],
#         tokens[-2] + vp_dir,
#         tokens[-1] + ".pkl.npy",
#     )
lidar_path = '/media/dell/hdd01/nuscenes/nuscenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin'
seg_path0 = '/media/dell/hdd01/nuscenes/virtual_points-MVP/virtual_points/samples/LIDAR_TOP_VIRTUAL/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin.pkl.npy'
            # media/dell/hdd01/nuscenes/nuscenes/virtual_points/samples/LIDAR_TOP_VIRTUAL/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin.pkl.npy
tokens = lidar_path.split("/")
vp_dir = "_VIRTUAL"
print(tokens)
seg_path = os.path.join(
        *tokens[:-4],
        "virtual_points-MVP",
        "virtual_points",
        tokens[-3],
        tokens[-2] + vp_dir,
        tokens[-1] + ".pkl.npy",
    )    
print(seg_path)