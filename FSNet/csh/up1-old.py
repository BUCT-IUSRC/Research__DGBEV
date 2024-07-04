import sys

import imageio
from PIL.Image import Image
from tqdm import tqdm

sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
# from vision_base.utils.builder import build
from vision_base.utils.builder import build
from vision_base.data.datasets.dataset_utils import collate_fn
from vision_base.utils.utils import cfg_from_file
import os
torch.cuda.set_device(0)
print('CUDA available: {}'.format(torch.cuda.is_available()))

#cfg = cfg_from_file("../configs/kitti360_gtpose_config.py")
#cfg = cfg_from_file("../configs/distill_monodepth2_gt_poseconfig.py")
cfg = cfg_from_file("/home/dell/csh/FSNet-master/configs/nuscenes_wpose.py")
#cfg = cfg_from_file("../configs/monodepth2_gtpose_uncertainty_config.py")
# cfg = cfg_from_file("../configs/kitti360_fisheye.py")
is_test_train = True

#checkpoint_name = "../workdirs/MonoDepth2_pose/checkpoint/MonoDepthWPose_ss11.pth"
#checkpoint_name = "../workdirs/Distillation_gtpose/checkpoint/DistillWPoseMeta_trained_ss8.pth"
#checkpoint_name = "../workdirs/MonoDepth2Nusc/checkpoint/MonoDepthWPose_ss11_threecam.pth"
checkpoint_name = "/data/csh_test/results/workdirs2/nusc_wpose/checkpoint/monodepth.networks.models.meta_archs.monodepth2_model.MonoDepthWPose_9.pth"
index = 0
split_to_test='training'
cfg.train_dataset.augmentation = cfg.val_dataset.augmentation
is_test_train = split_to_test == 'validation'
print(split_to_test)
if split_to_test == 'training':
    dataset = build(**cfg.train_dataset)
elif split_to_test == 'test':
    dataset = build(**cfg.test_dataset)

else:
    dataset  = build(**cfg.val_dataset)
    print(cfg.val_dataset)
meta_arch = build(**cfg.meta_arch)
meta_arch = meta_arch.cuda()
data_length=len(dataset)
weight_path = checkpoint_name
state_dict = torch.load(weight_path, map_location='cuda:{}'.format(cfg.trainer.gpu))
# /home/dell/csh/FSNet-master/resnet34-333f7ec4.pth
meta_arch.load_state_dict(state_dict['model_state_dict'])
meta_arch.eval();
test_hook = build(**cfg.trainer.evaluate_hook.test_run_hook_cfg)


def denorm(image):
    new_image = np.clip((image * cfg.data.augmentation.rgb_std + cfg.data.augmentation.rgb_mean) * 255, 0, 255)
    new_image = np.array(new_image, dtype=np.uint8)
    return new_image


from numba import jit


@jit(cache=False, nopython=True)
def ToColorDepth(depth_image: np.ndarray) -> np.ndarray:  # [H, W] -> [H, W, 3]
    H, W = depth_image.shape
    max_depth = float(np.max(depth_image))
    cmap = np.array([
        [0, 0, 0, 114], [0, 0, 1, 185], [1, 0, 0, 114], [1, 0, 1, 174],
        [0, 1, 0, 114], [0, 1, 1, 185], [1, 1, 0, 114], [1, 1, 1, 0]
    ])
    _sum = 0
    for i in range(8):
        _sum += cmap[i, 3]

    weights = np.zeros(8)
    cumsum = np.zeros(8)
    for i in range(7):
        weights[i] = _sum / cmap[i, 3]
        cumsum[i + 1] = cumsum[i] + cmap[i, 3] / _sum

    image = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            val = depth_image[i, j] / max_depth
            for k in range(7):
                if val <= cumsum[k + 1]:
                    break
            w = 1.0 - (val - cumsum[k]) * weights[k]
            r = int((w * cmap[k, 0] + (1 - w) * cmap[k + 1, 0]) * 255)
            g = int((w * cmap[k, 1] + (1 - w) * cmap[k + 1, 1]) * 255)
            b = int((w * cmap[k, 2] + (1 - w) * cmap[k + 1, 2]) * 255)
            image[i, j] = np.array([r, g, b])
    return image


def compute_once(index):
    data = dataset[index]
    collated_data = collate_fn([data])
    image = collated_data[('image', 0)]
    rgb_image = denorm(image[0].cpu().numpy().transpose([1, 2, 0]))
    with torch.no_grad():
        output_dict = test_hook(collated_data, meta_arch)
        depth = output_dict["depth"][0, 0]
        # print(depth.max(), depth.min())
        depth_uint16 = (depth * 256).cpu().numpy().astype(np.uint16)
        color_depth = ToColorDepth(depth_uint16)

    # plt.subplot(2, 2, 1)
    #
    # plt.imshow(np.clip(rgb_image, 0, 255)[:])
    # plt.subplot(2, 2, 2)
    # plt.imshow(1 / (depth_uint16 / 256)[:], cmap='magma', vmin=1.0 / (70), vmax=1 / max(depth_uint16.min() / 256, 2.0))
    #
    # plt.subplot(2, 2, 3)
    # color_depth = ToColorDepth(depth_uint16)
    # plt.imshow(depth_uint16 / 256)
    #
    # plt.subplot(2, 2, 4)
    # alpha = 0.3
    # masked = (alpha * color_depth + (1 - alpha) * rgb_image).astype(np.uint8)
    # plt.imshow(masked)
    #
    # plt.show()
    return np.clip(rgb_image, 0, 255), color_depth, depth_uint16

if __name__ == '__main__':

    out_dir = "/data/csh_test/vis_out/fsnet6"
    print(dataset[0][('filename', 0)])
    print(data_length)
    for i in (range(0,len(dataset))):
        # igigcs
        # i = 7 + 60

        rgb_image, color_depth, depth_uint16 = compute_once(i);

        path = dataset[i][('filename', 0)]
        parts = path.split("/")
        device = parts[1]
        pic_name = parts[2]
        # print(filename_index)

        dep900 = cv2.resize(depth_uint16, (1600, 900), interpolation=cv2.INTER_NEAREST)


        save_pic_path = os.path.join(out_dir, device, pic_name)
        save_pic_path=save_pic_path[:-3]+"png"
        dep900 = cv2.resize(depth_uint16, (1600, 900), interpolation=cv2.INTER_NEAREST)

        # 创建一个16位的PNG图像
        # image = Image.fromarray(dep900)

        # 保存图像为16位PNG文件
        imageio.imsave(save_pic_path, dep900)

        # image.save(save_pic_path, mode='I;16')

        print(i,"/",data_length)
        # print(save_pic_path)
        # break















