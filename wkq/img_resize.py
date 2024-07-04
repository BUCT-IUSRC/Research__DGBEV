from PIL import Image
'''
filein: 输入图片
fileout: 输出图片
width: 输出图片宽度
height:输出图片高度
type:输出图片类型（png, gif, jpeg...）
'''
def ResizeImage(filein, fileout, type):
  img = Image.open(filein)
  # img.show()
  out = img.resize((757, 425)) #resize image with high-quality
  out1 = out.crop((0, 106, 757, 425)).resize((704, 256))
  out1.show()
  out.save(fileout, type)
  out.show()
if __name__ == "__main__":
  filein = r'/home/dell/wkq/BEVFusion-mit/bevfusion-main/viz/camera-0/1531885320049418-88a090c54cc844cc878f00d08274a683.png'
  fileout = r'/home/dell/wkq/BEVFusion-mit/bevfusion-main/viz/camera-0/testout.png'

  type = 'png'
  ResizeImage(filein, fileout, type)

