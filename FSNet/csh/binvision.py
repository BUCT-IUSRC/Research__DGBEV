import numpy as np
import mayavi.mlab
import matplotlib.pyplot as plt
# lidar_path换成自己的.bin文件路径
import os
from PIL import Image
flnm=("/data/nuscenes/rgbd__image_test/depth/samples/CAM_FRONT/n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530412470.png")
depth_map = Image.open(flnm)
depth_map = np.array(depth_map)
print(np.max(depth_map),np.average((depth_map)))
for i in range(200):
    for j in range(200):
        if(depth_map[i][j]>0):
            print(depth_map[i][j])

plt.imshow( depth_map )
plt.show()
depth_map1 = Image.open((flnm))
depth_map1 = np.array(depth_map1)
print(np.max(depth_map1),np.average((depth_map1)))
for i in range(200):
    for j in range(200):
        if(depth_map1[i][j]>0):
            print(depth_map1[i][j])
plt.imshow( depth_map1 )
plt.show()
iguig



# 获取当前文件夹下所有文件名
file_list = os.listdir("/data/csh_test/vis_out/fsnet5/LIDAR_TOP")

# 打印文件名
x=file_list[6000]
print(x)
x="n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915255448617.pcd.bin.pkl.npy"
# pointcloud = np.load('/data/csh_test/vis_out/fsnet5/LIDAR_TOP/'+x, allow_pickle=True).T
pointcloud = np.load('/data/nuscenes/virtual_points-yolov5+depthmap/1/'+x, allow_pickle=True).T
# pointcloud = np.fromfile(('/home/dell/wkq/BEVFusion-mit/bevfusion-main/data-mini/nuscenes/samples/LIDAR_TOP/n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984245947391.pcd.bin'), dtype=np.float32, count=-1).reshape([-1, 5])
# pointcloud = np.fromfile(str(r'D:\wjdata\source3031\KITTI\object\testing\velodyne\000000.bin'), dtype=np.float32, count=-1).reshape([-1, 4])
print(pointcloud.shape)
x1 = pointcloud[:, 0]  # x position of point1
y1 = pointcloud[:, 1]  # y position of point
z1 = pointcloud[:, 2]  # z position of point
# print(x1.shape)

# virtual_points=points2['virtual_points'][:,:5]
# real_points=points2['real_points']
# real_points_indice=points2['real_points_indice']
# pointcloud=virtual_points

x = pointcloud[:, 0]  # x position of point1
y = pointcloud[:, 1]  # y position of point
z = pointcloud[:, 2]  # z position of point

# r = pointcloud[:, 3]  # reflectance value of point
d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

degr = np.degrees(np.arctan(z / d))

vals = 'height'
if vals == "height":
    col = z

else:
    col = d

fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1920, 1080))

mayavi.mlab.points3d(x, y, z,
                     col,  # Values used for Color
                     mode="point",
                     colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                     # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                     figure=fig,

                     )
mayavi.mlab.show()
