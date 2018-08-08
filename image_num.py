# src1 = '/Users/1-10robotics/Desktop/object_segmentation/models-master/research/deeplab/datasets/pepper_pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
# num_dict = {}
# cnt = 0
# with open(mode='r', file=src1) as f:
#   for line in f:
#     num_dict[cnt] = line.strip('\n')
#     cnt += 1
# # print(num_dict[71])


from PIL import Image
import numpy as np
import glob
import os

# pepper_pascal_voc_segのJPEGImages
pepper_pre_JPEGImages = '/Users/1-10robotics/Desktop/object_segmentation/models-master/research/deeplab/datasets/pepper_pascal_voc_seg/VOCdevkit/VOC2012/pre_JPEGImages'
# pepper_pascal_voc_segのpre_SegmentationClass
pepper_pre_SegmentationClass = '/Users/1-10robotics/Desktop/object_segmentation/models-master/research/deeplab/datasets/pepper_pascal_voc_seg/VOCdevkit/VOC2012/pre_SegmentationClass'
# pepper_pascal_voc_segのpre_SegmentationClassRaw
pepper_pre_SegmentationClassRaw = '/Users/1-10robotics/Desktop/object_segmentation/models-master/research/deeplab/datasets/pepper_pascal_voc_seg/VOCdevkit/VOC2012/pre_SegmentationClassRaw'
# pascal_voc_segのJPEGImages
pascal_JPEGImages = '/Users/1-10robotics/Desktop/object_segmentation/models-master/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/JPEGImages'

src_JPEGImages = '/Users/1-10robotics/Desktop/JPEGImages'
src_SegmentationClass = '/Users/1-10robotics/Desktop/SegmentationClass'
src_SegmentationClassRaw = '/Users/1-10robotics/Desktop/SegmentationClassRaw'

a = glob.glob("{}/*".format(pepper_pre_JPEGImages))
b = glob.glob("{}/*".format(pepper_pre_SegmentationClass))
c = glob.glob("{}/*".format(pepper_pre_SegmentationClassRaw))

for i in range(len(b)):
  name = os.path.basename(b[i])
  # 元となる画像の読み込み
  img = Image.open(b[i])
  #オリジナル画像の幅と高さを取得
  before_width, before_height = img.size
  if before_width > 500:
    after_width = 500
  else:
    after_width = before_width
  if before_height > 500:
    after_height = 500
  else:
    after_height = before_height

  if before_width != after_width or before_height != after_height:
    img_resize = img.resize((after_width, after_height))
    img_resize.save('{}/{}'.format(src_SegmentationClass, name))

max_width = 0
max_height = 0
width_dic = {}
height_dic = {}
width_dic['500<'] = 0
width_dic['<501'] = 0
width_dic['<401'] = 0
width_dic['<301'] = 0
width_dic['<201'] = 0
width_dic['<101'] = 0
height_dic['500<'] = 0
height_dic['<501'] = 0
height_dic['<401'] = 0
height_dic['<301'] = 0
height_dic['<201'] = 0
height_dic['<101'] = 0
for i in range(len(c)):
  # 元となる画像の読み込み
  img = Image.open(c[i])
  #オリジナル画像の幅と高さを取得
  width, height = img.size
  try:
    if width < 101:
      width_dic['<101'] += 1
    elif width < 201:
      width_dic['<201'] += 1
    elif width < 301:
      width_dic['<301'] += 1
    elif width < 401:
      width_dic['<401'] += 1
    elif width <501:
      width_dic['<501'] += 1
    else:
      width_dic['500<'] += 1
  except:
    pass
  try:
    if height < 101:
      height_dic['<101'] += 1
    elif height < 201:
      height_dic['<201'] += 1
    elif height < 301:
      height_dic['<301'] += 1
    elif height < 401:
      height_dic['<401'] += 1
    elif height < 501:
      height_dic['<501'] += 1
    else:
      height_dic['500<'] += 1
  except:
    pass

  if width > max_width:
    max_width = width
  if height > max_height:
    max_height = height
print(max_width)
print(max_height)
print(width_dic)
print(height_dic)
