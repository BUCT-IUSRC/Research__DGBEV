import torchvision
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
def img_visual(x):
    # [2,6,3,256,704]
    B, N, C, H, W = x.size()
    x = x.view(B * N, C, H, W)

    for i in range(B*N):
        print(type(x[i,:,:,:]))
        k = torchvision.transforms.ToTensor()
        to = torchvision.transforms.ToPILImage()
        new_img1 = to((x[i, :,:,:]))
        new_img1.show()
import mayavi.mlab
def points_visual(x):
    for i in range(len(x)):
        pointcloud=x[i]
        x = pointcloud[:, 0].cpu()  # x position of point1
        y = pointcloud[:, 1].cpu() # y position of point
        z = pointcloud[:, 2].cpu()  # z position of point
        d = np.sqrt((x ** 2 + y ** 2).cpu() ) # Map Distance from sensor
        col=d
        fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1920, 1080))
        mayavi.mlab.points3d(x, y, z,
                             col,  # Values used for Color
                             mode="point",
                             colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                             #  color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                             figure=fig,

                             )
