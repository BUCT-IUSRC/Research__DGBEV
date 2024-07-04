from random import sample
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='/home/dell/wkq/BEVFusion-mit/bevfusion-main/data/nuscenes', verbose=True)
nusc.list_scenes()
time_stamp='1533151603547590'
token1="3e8750f331d7499e9b5123e9eb70f2e2"
sample_tk=nusc.get('sample',token1)
# print(sample_tk)

# scene_token= "325cef682f064c55a255f2625c533b75"
sample=nusc.get('scene',sample_tk['scene_token'])
print(sample)
scene_name=sample['name']
print(scene_name)
# scene-0916
my_scene = nusc.scene[6]
my_sample={
"token": "8cd36e9531fb4eba8e6ac1d666c4641c",
"timestamp": 1538984246447815,
"prev": "44237858a539457da65822bfcf58c414",
"next": "7bebe3c9be714f02837f8617c56df122",
"scene_token": "325cef682f064c55a255f2625c533b75"
}

# my_sample = nusc.sample[10]
nusc.render_pointcloud_in_image(token1, pointsensor_channel='LIDAR_TOP')

