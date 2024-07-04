
import sys

import imageio
from PIL.Image import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
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
split_to_test='validation'
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
every_lenth=data_length//12

if __name__ == '__main__':

    out_dir = "/data/csh_test/vis_out/fsnet6"
    
    
    for i in tqdm(range(1*every_lenth,2*every_lenth)):
        # print(i," / ",every_lenth)
        data = dataset[i]
        collated_data = collate_fn([data])
        with torch.no_grad():
            output_dict = test_hook(collated_data, meta_arch)
            depth = output_dict["depth"][0, 0]
            # print(depth.max(), depth.min())
            depth_uint16 = (depth * 256).cpu().numpy().astype(np.uint16)
            path = dataset[i][('filename', 0)]
            parts = path.split("/")
            device = parts[1]
            pic_name = parts[2]
            dep900 = cv2.resize(depth_uint16, (1600, 900), interpolation=cv2.INTER_NEAREST)
            save_pic_path = os.path.join(out_dir, device, pic_name)
            save_pic_path=save_pic_path[:-3]+"png"
            # print(save_pic_path)
            imageio.imsave(save_pic_path, dep900)
            # igig













