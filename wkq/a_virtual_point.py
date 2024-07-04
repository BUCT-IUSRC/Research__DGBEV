import numpy as np
import os
import kitti_util

np.set_printoptions(threshold=np.inf)
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# 读取pcd点云文件
def read_pcd(filepath):
    lidar = []
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        while line:
            linestr = line.split(" ")
            if len(linestr) == 4:
                linestr_convert = list(map(float, linestr))
                lidar.append(linestr_convert)
            line = f.readline().strip()
    return np.array(lidar)

# 图像坐标系点反投影为点云
def img_to_rect(u, v, depth_rect):
    depth_cam_matrix = np.array([[855.14477539, 0.0, 960.90000463, 0.0], [0.0, 852.53753662, 551.05244629, 0.0], [0.0, 0.0, 1.0, 0.0]])
    fu, fv = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cu, cv = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
    tx, ty = depth_cam_matrix[0, 3] / (-fu), depth_cam_matrix[1, 3] / (-fv)
    x = ((u - cu) * depth_rect) / fu + tx
    y = ((v - cv) * depth_rect) / fv + ty
    pts_rect = np.concatenate((x.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32), depth_rect.reshape(-1, 1).astype(np.float32)), axis=1)
    pts_rect = calib.project_ref_to_velo(pts_rect)
    return pts_rect.astype(np.float32)

pc_path = r'D:/wjdata/source3031/pcd'
point_path = r'D:/wjdata/source3031/predicted_velodyne_test'
path = r'D:/wjdata/source3031/KITTI/object/testing/label_2'
calib_path = r'D:/wjdata/3000/timetb/calib1'
name_list = os.listdir(os.path.join(pc_path))
name_list.sort()

for name in name_list:
    virtual_pc = []
    pc = read_pcd(os.path.join(pc_path, name))
    pc_xyz = pc[:, 0:3]
    n = pc.shape[0]
    pc_xyz1 = np.hstack((pc_xyz, np.ones((n, 1))))
    T_cam0_velo = np.array([[-0.9988938, -0.04685274, -0.00399877, 0.064671524],
                            [-0.00182372, 0.12357472, -0.99233359, 0.075586899],
                            [0.04698769, -0.99122859, -0.12352346, -0.656047405]])
    pointcloud = np.dot(pc_xyz1, T_cam0_velo.T)
    n = pointcloud.shape[0]
    pointcloud1 = np.hstack((pointcloud, np.ones((n, 1))))
    P_rect_20 = np.array(
        [[855.14477539, 0.0, 960.90000463, 0.0], [0.0, 852.53753662, 551.05244629, 0.0], [0.0, 0.0, 1.0, 0.0]])
    point2d = np.array(np.dot(pointcloud1, P_rect_20.T))
    point2d[:, 0] /= point2d[:, 2]
    point2d[:, 1] /= point2d[:, 2]
    label = name[0:-3] + 'txt'
    with open(os.path.join(path, label)) as f:
        lines = f.readlines()
    for m in range(0, len(lines)):
        obj = lines[m].strip().split(' ')[4:8]
        # index_nozero = np.argwhere((point2d[:, 0] > float(obj[1])) & (point2d[:, 0] < float(obj[3])) & (point2d[:, 1] > float(obj[0])) & (
        #             point2d[:, 1] < float(obj[2])) & (point2d[:, 2]>0))
        index_nozero = np.argwhere((point2d[:, 0] > float(obj[0])) & (point2d[:, 0] < float(obj[2])) & (point2d[:, 1] > float(obj[1])) & (
                    point2d[:, 1] < float(obj[3])) & (point2d[:, 2]>0))
        # print('index_nozero:',len(index_nozero))
        # 随机点的选取
        all_points=[]
        for i in range(0,min(100,len(index_nozero))):
            generateddata=[np.random.randint(float(obj[0]),float(obj[2])),np.random.randint(float(obj[1]),float(obj[3]))]
            if not generateddata in all_points:
                all_points.append(generateddata)
        all_points = np.array(all_points)
        # print('all_points:',all_points)
        for j in range(0,all_points.shape[0]):
            dis_all = []
            for w in range(0,len(index_nozero)):
                dis = ((abs(point2d[index_nozero[w],0]-all_points[j,0]))**2 + (abs(point2d[index_nozero[w],1]-all_points[j,1])))**0.5
                dis_all.append(dis)
            # print('dis_all:',dis_all)
            index = np.argmin(dis_all)
            # print('index:',index)
            calib_file = os.path.join(calib_path, label)
            calib = kitti_util.Calibration(calib_file)
            x = img_to_rect(all_points[j,0],all_points[j,1],point2d[index_nozero[index],2])
            virtual_pc.append(x[0])
    virtual_pc = np.array(virtual_pc)
    virtual_pc = np.concatenate([virtual_pc, np.ones((virtual_pc.shape[0], 1))], 1)
    new_pc = np.concatenate((pc,virtual_pc),axis=0)
    print('pc:',new_pc.shape)
    save_name = name[0:-3] + 'bin'
    filename = os.path.join(point_path, save_name)
    pl = new_pc.reshape(-1, 4).astype(np.float32)
    rot = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])
    pl_xyz = np.array(np.dot(pl[:, 0:3], rot)).astype(np.float32)
    pl = np.concatenate([pl_xyz, pl[:, 3:4]], axis=1)
    lidar = pl.astype(np.float32)
    lidar.tofile(filename)
    print('Finish Depth {}'.format(filename))
print('-----end----')