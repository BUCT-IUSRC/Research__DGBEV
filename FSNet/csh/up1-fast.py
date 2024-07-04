"""
    Script for launching training process
./up1-fast.sh configs/nuscenes_wpose.py 0,1,2 fsnet7

"""
import matplotlib.pyplot as plt
import sys

import cv2
import imageio
import numpy as np
import os
import shutil
from easydict import EasyDict
from fire import Fire
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
sys.path.append("/home/dell/csh/FSNet-master/")
# from scripts._path_init import manage_package_logging

from vision_base.utils.builder import build
from vision_base.utils.utils import get_num_parameters, cfg_from_file, set_random_seed, update_cfg
from vision_base.utils.timer import Timer
from vision_base.utils.logger import LossLogger, styling_git_info
from vision_base.data.datasets.dataset_utils import collate_fn
from vision_base.data.dataloader import build_dataloader
from vision_base.networks.optimizers import optimizers, schedulers
from vision_base.networks.utils.utils import save_models, load_models

def main(config="configs/config.py", experiment_name="default", world_size=1, local_rank=-1, **kwargs):
    """Main function for the training script.

    KeywordArgs:
        config (str): Path to config file.
        experiment_name (str): Custom name for the experitment, only used in tensorboard.
        world_size (int): Number of total subprocesses in distributed training.
        local_rank: Rank of the process. Should not be manually assigned. 0-N for ranks in distributed training (only process 0 will print info and perform testing). -1 for single training.
    """

    ## Get config
    print("config")
    print(config)
    cfg = cfg_from_file("/home/dell/csh/FSNet-master/configs/nuscenes_wpose.py")
    print(cfg.path)
    cfg = update_cfg(cfg, **kwargs)

    ## Collect distributed(or not) information
    cfg.dist = EasyDict()
    cfg.dist.world_size = world_size
    cfg.dist.local_rank = local_rank
    is_distributed = local_rank >= 0 # local_rank < 0 -> single training
    is_logging     = local_rank <= 0 # only log and test with main process
    is_evaluating  = local_rank <= 0

    ## Setup writer if local_rank > 0
    recorder_dir = os.path.join(cfg.path.log_path, experiment_name + f"config={config}")
    if is_logging: # writer exists only if not distributed and local rank is smaller
        ## Clean up the dir if it exists before
        if os.path.isdir(recorder_dir):
            shutil.rmtree(recorder_dir, ignore_errors=True)
            print("clean up the recorder directory of {}".format(recorder_dir))
        writer = SummaryWriter(recorder_dir)

        ## Record config object using pprint
        import pprint

        formatted_cfg = pprint.pformat(cfg)
        writer.add_text("config.py", formatted_cfg.replace(' ', '&nbsp;').replace('\n', '  \n')) # add space for markdown style in tensorboard text

        ## Record Git status
        import git
        
        # print(cfg.path.base_path)

        repo = git.Repo(cfg.path.base_path)
        # repo.index.commit("Initial commit")

        writer.add_text("git/git_show", styling_git_info(repo))
        writer.flush()
    else:
        writer = None

    ## Set up GPU and distribution process
    if is_distributed:
        cfg.trainer.gpu = local_rank # local_rank will overwrite the GPU in configure file
    gpu = min(cfg.trainer.gpu, torch.cuda.device_count() - 1)
    torch.backends.cudnn.benchmark = getattr(cfg.trainer, 'cudnn', False)
    set_random_seed(123)
    torch.cuda.set_device(gpu)
    if is_distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    print(local_rank)

    ## Precomputing Hooks
    if 'precompute_hook' in cfg:
        precompute_hook = build(**cfg.precompute_hook)
        precompute_hook()
 
    ## define datasets and dataloader.
    dataset_train = build(**cfg.train_dataset)
    dataset_val = build(**cfg.val_dataset)

    dataloader_train = build_dataloader(dataset_train,
                                        num_workers=12,
                                        batch_size=24,
                                        collate_fn=collate_fn,
                                        local_rank=local_rank,
                                        world_size=world_size,
                                        sampler_cfg=getattr(cfg.data, 'sampler', dict()))

    ## Create the model
    meta_arch = build(**cfg.meta_arch)
    from vision_base.networks.models.meta_archs.base_meta import BaseMetaArch
    assert isinstance(meta_arch, BaseMetaArch)

    ## Convert to cuda
    if is_distributed:
        meta_arch = torch.nn.SyncBatchNorm.convert_sync_batchnorm(meta_arch)
        meta_arch = torch.nn.parallel.DistributedDataParallel(meta_arch.cuda(), device_ids=[gpu], output_device=gpu)
    else:
        meta_arch = meta_arch.cuda()
    checkpoint_name = "/data/csh_test/results/workdirs2/nusc_wpose/checkpoint/monodepth.networks.models.meta_archs.monodepth2_model.MonoDepthWPose_9.pth"
    weight_path = checkpoint_name
    load_models(weight_path, meta_arch, map_location=f'cuda:{gpu}', strict=False)
    meta_arch.eval();
    test_hook = build(**cfg.trainer.evaluate_hook.test_run_hook_cfg)
    ## Record basic information of the model

    
    ## define optimizer and weight decay
    optimizer = optimizers.build_optimizer(meta_arch, **cfg.optimizer)

    ## define scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.trainer.max_epochs, cfg.optimizer.lr_target)
    scheduler_config = getattr(cfg, 'scheduler', None)
    scheduler = schedulers.build_scheduler(optimizer, **scheduler_config)
    is_iter_based = getattr(scheduler_config, "is_iter_based", False)

    ## define loss logger
    training_loss_logger = LossLogger(writer, 'train') if is_logging else None
    # manage_package_logging()

    ## Load old model if needed
    old_checkpoint = getattr(cfg.path, 'pretrained_checkpoint', None)
    





    ## timer is used to estimate eta
    timer = Timer()

    print('Num training images: {}'.format(len(dataset_train)))

    global_step = 0
    out_dir = "/data/csh_test/vis_out/fsnet6"
    for epoch_num in range(1):
        ## Start training for one epoch
        meta_arch.eval()
        iter_num=0
        batch_size = 24
        depth_data = []
        paths = []
        for batched_data in tqdm(dataloader_train):
            
            # training_hook(batched_data, meta_arch, optimizer, writer, training_loss_logger, global_step, epoch_num)
            output_dict = test_hook(batched_data, meta_arch,global_step=global_step)
            B = output_dict['depth'].shape[0]
            for i in range(B):
                depth = output_dict["depth"][i, 0]
                h_eff, w_eff = batched_data[('image_resize', 'effective_size')][i]
                depth = depth[0:h_eff, 0:w_eff]
                # h, w, _ = batched_data[('original_image', 0)][i].shape
                dep900 = cv2.resize((depth * 256).cpu().detach().numpy().astype(np.uint16), (1600, 900))
                # print(np.max(dep900))
                # print(dep900.shape)
                path = batched_data['filename', 0][i]
                parts = path.split("/")
                device = parts[1]
                pic_name = parts[2]
                save_pic_path = os.path.join(out_dir, device, pic_name)
                save_pic_path = save_pic_path[:-3] + "png"
                # print(save_pic_path)
                imageio.imsave(save_pic_path, dep900)
                # plt.imshow(dep900)
                # plt.show()
                # iiff
            global_step += 1

 


  


if __name__ == '__main__':
    Fire(main)
