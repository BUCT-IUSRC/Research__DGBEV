from PIL import Image, ImageDraw, ImageFont
import os

# 输入路径A、B和输出路径C
path_0='/home/dell/wkq/BEVFusion-mit/run/full/baseline/gt/viz/lidar'
path_A = '/data/workdirs/bevfusion/depth_map5/bbs0.2/lidar'
path_B = '/home/dell/wkq/BEVFusion-mit/run/full/baseline/viz.bbs0.3/lidar/'
path_C = '/data/workdirs/bevfusion/depth_map5/bbs0.3_mit/gt_7122_7187/lidar/'

# 获取路径A和B中的所有图片文件名
files_GT = os.listdir(path_0)
files_A = os.listdir(path_A)
files_B = os.listdir(path_B)

# 确保路径C存在
if not os.path.exists(path_C):
    os.makedirs(path_C)

# 遍历路径A中的文件
for file_A in files_B:
    # 检查文件是否存在于路径B中
    if file_A in files_B and file_A in files_GT:
        # 打开路径A和B中的图片
        img_GT = Image.open(os.path.join(path_0, file_A))
        img_A = Image.open(os.path.join(path_A, file_A))
        img_B = Image.open(os.path.join(path_B, file_A))

        # 创建一个新的图像，将两个图片并排拼接
        new_img = Image.new('RGB', (img_A.width+img_A.width + img_B.width, max(img_A.height, img_B.height)))
        new_img.paste(img_GT, (0, 0))
        new_img.paste(img_B, (img_A.width, 0))
        new_img.paste(img_A, (img_A.width+img_A.width, 0))


        # 保存新图像到路径C
        new_img.save(os.path.join(path_C, file_A))

        # 关闭打开的图片
        img_GT.close()
        img_A.close()
        img_B.close()

print("操作完成")
