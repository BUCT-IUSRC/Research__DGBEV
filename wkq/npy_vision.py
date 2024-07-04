import numpy as np
import mayavi.mlab

# lidar_path换成自己的.bin文件路径
pointcloud = np.fromfile(str(r'/media/dell/hdd01/nuscenes/nuscenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin'), dtype=np.float32, count=-1).reshape([-1, 5])
# pointcloud = np.fromfile(str(r'D:\wjdata\source3031\KITTI\object\testing\velodyne\000000.bin'), dtype=np.float32, count=-1).reshape([-1, 4])
x1 = pointcloud[:, 0]  # x position of point1
y1 = pointcloud[:, 1]  # y position of point
z1 = pointcloud[:, 2]  # z position of point

points2 = np.load('/media/dell/hdd01/nuscenes/virtual_points-MVP/virtual_points/samples/LIDAR_TOP_VIRTUAL/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin.pkl.npy', allow_pickle=True).item()
virtual_points=points2['virtual_points'][:,:3]
real_points=points2['real_points']
real_points_indice=points2['real_points_indice']

pointcloud=virtual_points
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
fig1 = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1920, 1080))
fig2 = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1920, 1080))
mayavi.mlab.points3d(x, y, z,
                     col,  # Values used for Color
                     mode="point",
                     colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                     # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                     figure=fig,
                     
                     )

mayavi.mlab.points3d(x1, y1, z1,
                     z1,  # Values used for Color
                     mode="point",
                     colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                     # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                     figure=fig1,
                     )

x3=[]
for i in x1:
    x3.append(i)
for i in x:
        x3.append(i)             
y3=[]
for i in y1:
    y3.append(i)
for i in y:
        y3.append(i)            
z3=[]
for i in z1:
    z3.append(i)
for i in z:
        z3.append(i)            
 
mayavi.mlab.points3d(x3, y3, z3,
                     z3,  # Values used for Color
                     mode="point",
                     colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                     # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                     figure=fig2,
                     )                
mayavi.mlab.show()
