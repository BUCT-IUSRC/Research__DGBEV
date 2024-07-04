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
from torch.utils.data import DataLoader

cfg = cfg_from_file("/home/dell/csh/FSNet-master/configs/nuscenes_wpose.py")

is_test_train = True

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

# meta_arch = torch.nn.SyncBatchNorm.convert_sync_batchnorm(meta_arch)
# meta_arch = torch.nn.parallel.DistributedDataParallel(meta_arch.cuda(), device_ids=[0,1,2],output_device=[0,1,2])
data_length=len(dataset)
print(data_length)
weight_path = checkpoint_name
cfg.trainer.gpu=0
state_dict = torch.load(weight_path, map_location='cuda:{}'.format(cfg.trainer.gpu))
# /home/dell/csh/FSNet-master/resnet34-333f7ec4.pth
meta_arch.load_state_dict(state_dict['model_state_dict'])
meta_arch.eval();
test_hook = build(**cfg.trainer.evaluate_hook.test_run_hook_cfg)
if __name__ == '__main__':

    out_dir = "/data/csh_test/vis_out/fsnet7"
    dataloader = DataLoader(dataset, batch_size=24, shuffle=False, num_workers=6,
                            collate_fn=collate_fn)
    for batched_data in tqdm(dataloader):
        output_dict = test_hook(batched_data, meta_arch)
        B = output_dict['depth'].shape[0]
        for i in range(B):
            depth = output_dict["depth"][i, 0]
            depth_uint16 = (depth * 256).cpu().detach().numpy().astype(np.uint16)
            dep900 = cv2.resize(depth_uint16, (1600, 900), interpolation=cv2.INTER_NEAREST)
            # print(batched_data['filename', 0][i])
            path = batched_data['filename', 0][i]
            parts = path.split("/")
            device = parts[1]
            pic_name = parts[2]
            save_pic_path = os.path.join(out_dir, device, pic_name)
            save_pic_path=save_pic_path[:-3]+"png"
            # print(save_pic_path)
            imageio.imsave(save_pic_path, dep900)
