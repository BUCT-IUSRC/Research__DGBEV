import numpy as np
import mayavi.mlab

# lidar_path换成自己的.bin文件路径
# pointcloud = np.load('/data/nuscenes/virtual_points-yolov5+depthmap/1/n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984242947023.pcd.bin.pkl.npy')
# pointcloud = np.fromfile(('/home/dell/wkq/BEVFusion-mit/bevfusion-main/data-mini/nuscenes/samples/LIDAR_TOP/n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984245947391.pcd.bin'), dtype=np.float32, count=-1).reshape([-1, 5])
pointcloud = np.fromfile(str(r'/home/dell/Desktop/wkq/lucky/rgb/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915255448617.pcd.bin'), dtype=np.float32, count=-1).reshape([-1, 5])
# x1 = pointcloud[:, 0]  # x position of point1
# y1 = pointcloud[:, 1]  # y position of point
# z1 = pointcloud[:, 2]  # z position of point
# print(x1.shape)
# points2 = np.load('/data/virtual_points-MVP/virtual_points/samples/LIDAR_TOP_VIRTUAL/n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984245947391.pcd.bin.pkl.npy', allow_pickle=True).item()
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
    # col= pointcloud[:, 3]
else:
    col = d

fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1920, 1080))
# fig1 = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1920, 1080))
# fig2 = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1920, 1080))
mayavi.mlab.points3d(x, y, z,
                     col,  # Values used for Color
                     mode="point",
                     colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                     # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                     figure=fig,

                     )
#
# mayavi.mlab.points3d(x1, y1, z1,
#                      z,  # Values used for Color
#                      mode="point",
#                      colormap='spectral',  # 'bone', 'copper', 'gnuplot'
#                      # color=(0, 0, 1),   # Used a fixed (r,g,b) instead
#                      figure=fig1,
#                      )

# x3=[]
# for i in x1:
#     x3.append(i)
# for i in x:
#         x3.append(i)             
# y3=[]
# for i in y1:
#     y3.append(i)
# for i in y:
#         y3.append(i)            
# z3=[]
# for i in z1:
#     z3.append(i)
# for i in z:
#         z3.append(i)            
 
# mayavi.mlab.points3d(x3, y3, z3,
#                      z3,  # Values used for Color
#                      mode="point",
#                      colormap='spectral',  # 'bone', 'copper', 'gnuplot'
#                      # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
#                      figure=fig2,
#                      )
mayavi.mlab.show()
