import torch
import open3d as o3d
import os
import os.path as osp
import numpy as np
import struct
import open3d
import mayavi.mlab
from tkinter import NONE
from typing import NoReturn
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import time


# from depth2 import *



def get_sensor_from_img_filepath(depth_root, i, filename, nusc, root):
    # print(root)
    # print(filename)
    index = filename.find("samples/")
    if (index != -1):
        filename = filename[index:]
    print(filename)
    token = nusc.field2token('sample_data', 'filename', filename)
    print(token)
    # samples/CAM_FRONT/n008-2018-08-30-15-16-55-0400__CAM_FRONT__1535657119612404.jpg
    sample_data_token = nusc.get('sample_data', token[0])
    # print(sample_data_token)
    sample_record = nusc.get('sample', sample_data_token['sample_token'])
    pointsensor_channel = 'LIDAR_TOP'
    first_index = filename.find("/") + 1
    second_index = filename.find("/", first_index + 1)
    # print(first_index,second_index)
    if (first_index != -1 and second_index != -1):
        camera_channel = filename[first_index:second_index]
    pointsensor_token = sample_record['data'][pointsensor_channel]
    camera_token = sample_record['data'][camera_channel]
    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
    pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
    print(pcl_path)
    if pointsensor['sensor_modality'] == 'lidar':
        pc = LidarPointCloud.from_file(pcl_path)
    else:
        pc = RadarPointCloud.from_file(pcl_path)
    org_pc=pc.points[:3,:]
    points_vis_one(org_pc.T)
    print(org_pc.shape)
    cs_record1 = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])

    poserecord1 = nusc.get('ego_pose', pointsensor['ego_pose_token'])

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord2 = nusc.get('ego_pose', cam['ego_pose_token'])

    # Fourth step: transform from ego into the camera.
    cs_record2 = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    depmap_dir=depth_root + "/" + i
    depth_map = Image.open(depmap_dir)
    print(depmap_dir)
    plt.imshow(depth_map)
    plt.show()
    # show_dep_comp_map(depth_map)
    # 定义一个转换函数，用于将图像转换为张量
    transform = transforms.ToTensor()

    # 使用转换函数将图像转换为张量
    depth_map = transform(depth_map)

    height = depth_map.shape[-2]
    width = depth_map.shape[-1]
    depth_map = depth_map.view(-1)
    nonzero_indices = torch.nonzero(depth_map<8000).squeeze()
    depth_map = depth_map[nonzero_indices].squeeze()
    y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width))
    y_coords = y_coords.reshape(-1)[nonzero_indices].squeeze()
    x_coords = x_coords.reshape(-1)[nonzero_indices].squeeze()
    camera_coords = torch.stack((x_coords, y_coords, torch.ones_like(x_coords)), dim=-1).float()
    nbr_points = camera_coords.shape[0]
    print(camera_coords.shape)
    depth_map_reshaped = depth_map.reshape(-1, 1)
    points = camera_coords * depth_map_reshaped
    points = points.T  # 3*n
    depth_map=Image.open(depth_root+"/"+i)
        # 定义一个转换函数，用于将图像转换为张量
    transform = transforms.ToTensor()

    # 使用转换函数将图像转换为张量
    depth_map = transform(depth_map)

    height = depth_map.shape[-2]
    width = depth_map.shape[-1]
    depth_map = depth_map.view(-1)
    nonzero_indices = torch.nonzero(depth_map)
    depth_map = depth_map[nonzero_indices].squeeze()
    y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width))
    y_coords = y_coords.reshape(-1)[nonzero_indices].squeeze()
    x_coords = x_coords.reshape(-1)[nonzero_indices].squeeze()
    camera_coords = torch.stack((x_coords, y_coords, torch.ones_like(x_coords)), dim=-1).float()
    nbr_points = camera_coords.shape[0]
    print(camera_coords.shape)
    depth_map_reshaped = depth_map.reshape(-1, 1)
    points = camera_coords * depth_map_reshaped
    points = points.T  # 3*n
    view = np.array(cs_record2['camera_intrinsic'])
    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view
    # print(viewpad)
    points = np.concatenate((points, np.ones((1, nbr_points))))  # 4*n
    points = np.dot(np.linalg.inv(viewpad), points)
    points = points[:3, :]
    pc.points = points
    # print(points.shape)

    pc.rotate(np.linalg.inv(Quaternion(cs_record2['rotation']).rotation_matrix.T))
    pc.translate(np.array(cs_record2['translation']))
    # print(pc.points[:3, :10]) # test_4

    # The reverse process of 3 step
    pc.rotate(np.linalg.inv(Quaternion(poserecord2['rotation']).rotation_matrix.T))
    pc.translate(np.array(poserecord2['translation']))
    # print(pc.points[:3, :10]) # test_3

    # The reverse process of 2 step
    pc.translate(-np.array(poserecord1['translation']))
    pc.rotate(np.linalg.inv(Quaternion(poserecord1['rotation']).rotation_matrix))
    # print(pc.points[:3, :10]) # test_2

    # The reverse process of 1 step
    pc.translate(-np.array(cs_record1['translation']))
    pc.rotate(np.linalg.inv(Quaternion(cs_record1['rotation']).rotation_matrix))
    virtual_points = pc.points

    all_points=np.concatenate((virtual_points,org_pc),axis=1)
    print("all_points.shape")
    print(all_points.shape)
    points_vis_one(all_points.T)
    return virtual_points



def show_dep_comp_map(map):

            plt.imshow(map)
            plt.show()


def points_vis_one(geom1):
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1920, 1080))
    mayavi.mlab.points3d(geom1[:, 0], geom1[:, 1], geom1[:, 2],
                         geom1[:, 2],  # Values used for Color
                         mode="point",
                         colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )
    mayavi.mlab.show()

def get_filename(depth_root):
    filename=os.listdir(depth_root)
    print(len(filename))
    return filename

if __name__ == '__main__':
    print("start")
    start_time = time.time()
    # file_path = r'/home/dell/csh/code_uploade/metas/metas_bs2.pt'
    # root = r'/home/dell/wkq/BEVFusion-mit/bevfusion-main'
    root = "data/nuscenes/nuscenes-full/"
    nusc_root = '/data/nuscenes/nuscenes-full'
    nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_root)
    depth_root="/data/csh_test/vis_out/fsnet1/CAM_FRONT"
    filename=get_filename(depth_root)
    # print(images.shape)
    # torch.Size([bs, 6, 3, 900, 1600])

    # step 1 : lidar to pic ,get depth_map
    nusc_root = '/data/nuscenes/nuscenes-full'
    nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_root)
    # print("start")
    for i in filename:
        print(i)
        # n015-2018-07-24-10-42-41+0800__CAM_BACK__1532400530637525.png

        # print(i)
        image_root = os.path.join("samples", "CAM_FRONT", i)[:-3] + "jpg"
        print(image_root)
        # samples/CAM_BACK/n015-2018-07-24-10-42-41+0800__CAM_BACK__1532400530637525.png

        vp= get_sensor_from_img_filepath(depth_root, i, image_root,  nusc, root)
        break

    # points_vis_one(vp.T)
    end_time = time.time()

    print("running for .......", (end_time - start_time))
