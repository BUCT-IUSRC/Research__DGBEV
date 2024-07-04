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
import matplotlib.pyplot  as plt
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
# from depth2 import
import time
def read_from_file(file_path):
    metas = torch.load(file_path, map_location=torch.device('cuda'))
    return metas
def analysis_metas(metas):
    filenames = []


    for meta in metas:
        filenames.append(meta["filename"])


    return filenames


def filename_to_image(filename, root):
    batch_size = len(filename)
    images_bs = []
    for bs in range(batch_size):
        images = []
        for i in filename[bs]:
            if len(root) > 1:
                i = root + i[7:]
            im = Image.open(i)
            image = transforms.ToTensor()(im)
            images.append(image)
        tensor = torch.stack(images, dim=0)
        tensor = tensor.unsqueeze(0)
        images_bs.append(tensor)
    tensor_bs = torch.stack(images_bs, dim=1)
    tensor_bs = tensor_bs.squeeze(0).to('cuda')
    return tensor_bs


def get_sensor_from_img_filepath(filename,nusc,root):

    index=filename.find("samples/")
    if(index!= -1):
        filename=filename[index:]

    token = nusc.field2token('sample_data', 'filename', filename)
    sample_data_token = nusc.get('sample_data', token[0])
    # print(sample_data_token)
    sample_record = nusc.get('sample', sample_data_token['sample_token'])
    pointsensor_channel = 'LIDAR_TOP'
    first_index=filename.find("/")+1
    second_index = filename.find("/",first_index+1)
    # print(first_index,second_index)
    if(first_index !=-1 and second_index!=-1):
        camera_channel = filename[first_index:second_index]
    pointsensor_token = sample_record['data'][pointsensor_channel]
    camera_token = sample_record['data'][camera_channel]
    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
    pcl_path =osp.join(nusc.dataroot, pointsensor['filename'])

    im = Image.open(osp.join(nusc.dataroot, cam['filename']))

    if pointsensor['sensor_modality'] == 'lidar':
        pc = LidarPointCloud.from_file(pcl_path)
    else:
        pc = RadarPointCloud.from_file(pcl_path)
    cs_record1 = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record1['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record1['translation']))
    # Second step: transform from ego to the global frame.
    poserecord1 = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    # print(pc.points[:3, :10]) # test_2
    pc.rotate(Quaternion(poserecord1['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord1['translation']))
    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord2 = nusc.get('ego_pose', cam['ego_pose_token'])
    # print(pc.points[:3, :10]) # test_3
    pc.translate(-np.array(poserecord2['translation']))
    pc.rotate(Quaternion(poserecord2['rotation']).rotation_matrix.T)
    # Fourth step: transform from ego into the camera.
    cs_record2 = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    # print(pc.points[:3, :10]) # test_4
    pc.translate(-np.array(cs_record2['translation']))
    pc.rotate(Quaternion(cs_record2['rotation']).rotation_matrix.T)
    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    # print(pc.points[:3, :10]) # test_5
    points = view_points(pc.points[:3, :], np.array(cs_record2['camera_intrinsic']), normalize=True)
    min_dist = 1.0
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    depths = depths[mask]
    im_width,im_height=im.size
    # print(im_width,im_height)
    dep_map = torch.zeros((1, im_height, im_width), device="cuda")
    for pnti in range(points.shape[1]):
        x_pnti = int(points[0, pnti])
        y_pnti = int(points[1, pnti])
        dep_pnti = depths[pnti]
        if dep_pnti > 0:
            dep_pnti=torch.tensor(dep_pnti)
            dep_map[0, y_pnti, x_pnti] = dep_pnti

    return dep_map,cs_record2,poserecord2,poserecord1,cs_record1,pc


def depth_comp_easy1(img, data):

    # zjy
    current_device = torch.cuda.current_device()

    depth_all = []
    for b in range(data.shape[0]):
        depth = []
        for iii in range(data.shape[1]):
            image = data[b, iii, :, :, :]
            # print(image.shape)
            # [1, 900, 1600]

            x = image.shape[-1]
            datai = data[b, iii, :, :, :].reshape(-1, x).cuda()
            # print(datai.shape)
            # [  900, 1600]
            # plt.imshow(datai,cmap='jet')
            # plt.show()

            depth_org = torch.stack([datai], dim=-1)
            # plt.imshow(depth_org)
            # plt.show()
            r = depth_org[:, :, 0]
            new_r = r
            col = depth_org.shape[0]
            row = depth_org.shape[1]
            nonzero_index = torch.nonzero(r > 0)
            # print (nonzero_index.shape)
            h_min = nonzero_index[:, 0].min().item()
            i = h_min
            i_l = 45
            j_l = 80
            while i < col:
                j = 0
                while j < row:
                    depth_block = depth_org[i:min(i + i_l, col), j:min(j + j_l, row)]
                    r_b = depth_block[:, :, 0]
                    nonzero_index_block = torch.nonzero(r_b > 0)
                    zero_index_block = torch.nonzero(r_b == 0)
                    if nonzero_index_block.shape[0] == 0 or zero_index_block.shape[0] == 0:
                        j = j + j_l
                        continue

                    nonzero_index_block1 = nonzero_index_block.unsqueeze(0)
                    nonzero_index_block1 = nonzero_index_block1.repeat(zero_index_block.shape[0], 1, 1)
                    zero_index_block1 = zero_index_block.unsqueeze(1)
                    dis = torch.abs(nonzero_index_block1 - zero_index_block1)
                    dis_sum = torch.sum(dis, dim=2)
                    index = torch.argmin(dis_sum, dim=1)

                    nonindex0, nonindex1 = nonzero_index_block[index][:, 0], nonzero_index_block[index][:, 1]
                    zeroindex0, zeroindex1 = zero_index_block[:, 0], zero_index_block[:, 1]
                    new_r[zeroindex0 + i, zeroindex1 + j] = r_b[nonindex0, nonindex1]

                    j = j + j_l
                i = i + i_l

            if (iii == 0):
                new_rr = new_r.reshape(1, 1, 1, -1, row)
            else:
                new_rr = torch.cat([new_rr, new_r.reshape(1, 1, 1, -1, row)], dim=1)

        depth_all.append(new_rr)
    depth_all = torch.stack(depth_all, dim=0)
    depth_all = depth_all.squeeze(dim=1)
    return depth_all

def get_dep_maps(filename,nusc,root):
    # 1,900,1600
    batch_size = len(filename)
    dep_map_one_bs=[]
    dep_map_all_bs = []
    cs_record2_all=[]
    poserecord2_all=[]
    poserecord1_all=[]
    cs_record1_all=[]
    for bs in range(batch_size):
        dep_map_one_bs = []
        cs_record2_one = []
        poserecord2_one = []
        poserecord1_one = []
        cs_record1_one= []
        for flnm in filename[bs]:
            dep_map,cs_record2,poserecord2,poserecord1,cs_record1,pc=get_sensor_from_img_filepath(flnm,nusc,root)
            # print(dep_map.shape)
            # dep_map=torch.from_numpy(dep_map).cuda()
            dep_map_one_bs.append(dep_map)
            cs_record2_one.append(cs_record2)
            poserecord2_one.append(poserecord2)
            poserecord1_one.append(poserecord1)
            cs_record1_one.append(cs_record1)
        # cs_record2_one = torch.stack(cs_record2_one, dim=0)
        cs_record2_all.append(cs_record2_one)
        poserecord2_all.append(poserecord2_one)
        poserecord1_all.append(poserecord1_one)
        cs_record1_all.append(cs_record1_one)
        dep_map_one_bs=torch.stack(dep_map_one_bs, dim=0)
        dep_map_all_bs.append(dep_map_one_bs)
    dep_map_all_bs = torch.stack(dep_map_all_bs, dim=0)
    # print(len(cs_record1_all))
    # print((cs_record1_all))
    # print(dep_map_all_bs.shape)
    # torch.Size([2, 6, 1, 900, 1600])
    return dep_map_all_bs,cs_record1_all,cs_record2_all,poserecord1_all,poserecord2_all,pc
    # torch.Size([2, 6, 1, 900, 1600])
    # tensor_bs = torch.stack(dep_maps, dim=0)
    # tensor_bs = tensor_bs.squeeze(0)
    # print(tensor_bs.shape)

def show_dep_comp_map(map):
    for b in range(map.shape[0]):
        for i in range(map.shape[1]):
            map_i=map[b,i,0,:,:]
            plt.imshow(map_i)
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

def dep_comp_maps2lidarpoint(dep_map_all_bs,cs_record1_all,cs_record2_all,poserecord1_all,poserecord2_all,pc):
    # print("dep_comp_maps2lidarpoint")
    # print(dep_map_all_bs.shape)
    # torch.Size([2, 6, 1, 900, 1600])
    batch_size=dep_map_all_bs.shape[0]
    virtual_points_all=[]
    for b in range(batch_size):
        # depth 2 depth_points
        # 1 6 1 256 704 to 1 6 n 2
        virtual_points_nv = np.empty(shape=(3,1))
        for n_views in range(dep_map_all_bs.shape[1]):
            cs_record1, cs_record2, poserecord1, poserecord2=cs_record1_all[b][n_views],cs_record2_all[b][n_views],poserecord1_all[b][n_views],poserecord2_all[b][n_views]
            depth_map = dep_map_all_bs[b, n_views, 0, :, :].float()  # 256,704
            # 找到深度图中非零点的索引
            height = depth_map.shape[-2]
            width = depth_map.shape[-1]
            depth_map = depth_map.view(-1)
            nonzero_indices = torch.nonzero(depth_map)
            depth_map = depth_map[nonzero_indices].squeeze()
            y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width))
            y_coords = y_coords.reshape(-1)[nonzero_indices].squeeze()
            x_coords = x_coords.reshape(-1)[nonzero_indices].squeeze()
            camera_coords=torch.stack((x_coords,y_coords,torch.ones_like(x_coords)),dim=-1).float()
            nbr_points = camera_coords.shape[0]
            # print(camera_coords.shape)
            # torch.Size([2954, 3])
            # print(depth_map.shape)
            # torch.Size([2954])
            depth_map_reshaped=depth_map.reshape(-1, 1)
            points = camera_coords.cpu() * depth_map_reshaped.cpu()
            points=points.T# 3*n

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
            # print(virtual_points.shape)
            # print(virtual_points_nv.shape)
            virtual_points_nv=np.concatenate((virtual_points,virtual_points_nv),axis=1)
            # print(virtual_points.shape)
        # points_vis_one(virtual_points_nv.T)
            # (3, 2954)
        virtual_points_all.append(virtual_points_nv)



    return virtual_points_all


def from_metas_get_vir_points(metas,nusc):
    # print("start")
    # file_path=r'/home/dell/csh/depth/matrix2/metas_bs2.pt'
    # # root = r'/home/dell/wkq/BEVFusion-mit/bevfusion-main'
    root = ""
    # metas=read_from_file(file_path)
    start_time1 = time.time()
    filename=analysis_metas(metas)
    print("in_current_time1:", time.time() - start_time1)
    images=filename_to_image(filename,root)
    print("in_current_time2:", time.time() - start_time1)
    # print(images.shape)
    # torch.Size([bs, 6, 3, 900, 1600])

    # step 1 : lidar to pic ,get depth_map
    # nusc_root = '/media/dell/hdd01/nuscenes/nuscenes'
    # nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_root)
    # print("start")
    depth_maps,cs_record1_all,cs_record2_all,poserecord1_all,poserecord2_all,pc=get_dep_maps(filename,nusc,root)
    print("in_current_time3:", time.time() - start_time1)
    dep_comp_map=depth_comp_easy1(images,depth_maps)
    print("in_current_time4:", time.time() - start_time1)
    # torch.Size([2, 6, 1, 900, 1600])
    # show_dep_comp_map(dep_comp_map)
    #step 2 : depth comp_map 2 point
    virtual_points_all=dep_comp_maps2lidarpoint(dep_comp_map,cs_record1_all,cs_record2_all,poserecord1_all,poserecord2_all,pc)
    print("in_current_time5:", time.time() - start_time1)
    return virtual_points_all

