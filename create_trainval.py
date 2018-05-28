import cv2
import glob
import os
import re
from collections import defaultdict

"""新規で画像を加える場合
WORK_DIR : VOC2012/
OUTPUT :
ImageSets/Main/trainval.txt
ImageSets/Main/pepper_trainval.txt
ImageSets/Segmentation/trainval.txt

画像ファイル一覧を取ってきて，ファイル名から生成する

trainとvalに分割する

splits_to_sizes={
    'train': 1464,
    'trainval': 2913,
    'val': 1449,
},
"""
# ディレクトリ内のファイル取得
CURRENT_DIR = os.getcwd()
PEPPER_IMG_DIR = os.path.join(CURRENT_DIR, 'pepper', 'images')
IMAGESET_DIR = os.path.join(CURRENT_DIR, 'ImageSets', 'Main')
SEGMENT_DIR = os.path.join(CURRENT_DIR, 'ImageSets', 'Segmentation')

# ファイルの行数を取得する
def read_n(path):
  with open(path, "r", encoding="utf-8") as f:
    lines = []
    for line in f:
          lines.append(line.rstrip('\r\n'))
  return lines

# フォルダ内のファイル数（.jpg）取得
def get_file_name(path):
  return glob.glob(path+'/*.jpg')

# 現在ある行数（訓練標本数）を取得し，末尾にpepper画像のファイル名を追加する
pepper_files = get_file_name(PEPPER_IMG_DIR)
trainval_images_list = read_n(IMAGESET_DIR+'/trainval.txt') + pepper_files
trainval_segment_list = read_n(SEGMENT_DIR+'/trainval.txt') + pepper_files
split = 0.8
train_images_list = read_n(IMAGESET_DIR+'/train.txt') + pepper_files[:int(len(pepper_files)*0.8)]
train_segment_list = read_n(SEGMENT_DIR+'/train.txt') + pepper_files[:int(len(pepper_files)*0.8)]
val_images_list = read_n(IMAGESET_DIR+'/val.txt') + pepper_files[int(len(pepper_files)*0.8):]
val_segment_list = read_n(SEGMENT_DIR+'/val.txt') + pepper_files[int(len(pepper_files)*0.8):]

print('●pascal')
print('images_train_val : {}'.format(len(read_n(IMAGESET_DIR+'/trainval.txt'))))
print('images_train : {}'.format(len(read_n(IMAGESET_DIR+'/train.txt'))))
print('images_val : {}'.format(len(read_n(IMAGESET_DIR+'/val.txt'))))
print('segment_train_val : {}'.format(len(read_n(SEGMENT_DIR+'/trainval.txt'))))
print('segment_train : {}'.format(len(read_n(SEGMENT_DIR+'/train.txt'))))
print('segment_val : {}'.format(len(read_n(SEGMENT_DIR+'/val.txt'))))
print(' ')
print('●pascal + pepper')
print('images_train_val : {}'.format(len(trainval_images_list)))
print('images_train : {}'.format(len(train_images_list)))
print('images_val : {}'.format(len(val_images_list)))
print('segment_train_val : {}'.format(len(trainval_segment_list)))
print('segment_train : {}'.format(len(train_segment_list)))
print('segment_val : {}'.format(len(val_segment_list)))


#書き込み用のリスト
trainval_images = [] #ImageSets/mainのtrainval.txt用
train_images = [] #ImageSets/mainのtrain.txt用
val_images = [] #ImageSets/mainのval.txt用
pepper_trainval = [] #ImageSets/main用
pepper_train = [] #ImageSets/main用
pepper_val = [] #ImageSets/main用

# trainval
for fname in trainval_images_list:
  fname = fname.split('/')[-1].split('.')[0]
  trainval_images.append(fname)
  if 'pepper' in fname:
    pepper_trainval.append(fname+" "+str(1))
  else:
    pepper_trainval.append(fname+" "+str(-1))

# train
for fname in train_images_list:
  fname = fname.split('/')[-1].split('.')[0]
  train_images.append(fname)
  if 'pepper' in fname:
    pepper_train.append(fname+" "+str(1))
  else:
    pepper_train.append(fname+" "+str(-1))

# val
for fname in val_images_list:
  fname = fname.split('/')[-1].split('.')[0]
  val_images.append(fname)
  if 'pepper' in fname:
    pepper_val.append(fname+" "+str(1))
  else:
    pepper_val.append(fname+" "+str(-1))

trainval_segment = [] #segmentationのtrainval.txt用
train_segment = [] #segmentationのtrainval.txt用
val_segment = [] #segmentationのtrainval.txt用
for fname in trainval_segment_list:
  fname = fname.split('/')[-1].split('.')[0]
  trainval_segment.append(fname)

for fname in train_segment_list:
  fname = fname.split('/')[-1].split('.')[0]
  train_segment.append(fname)

for fname in val_segment_list:
  fname = fname.split('/')[-1].split('.')[0]
  val_segment.append(fname)

_trainval_images = os.path.join(IMAGESET_DIR,"_trainval.txt")
_trainval_segmentation = os.path.join(SEGMENT_DIR,"_trainval.txt")
_pepper_trainval = os.path.join(IMAGESET_DIR,"_pepper_trainval.txt")
_train_images = os.path.join(IMAGESET_DIR,"_train.txt")
_train_segmentation = os.path.join(SEGMENT_DIR,"_train.txt")
_pepper_train = os.path.join(IMAGESET_DIR,"_pepper_train.txt")
_val_images = os.path.join(IMAGESET_DIR,"_val.txt")
_val_segmentation = os.path.join(SEGMENT_DIR,"_val.txt")
_pepper_val = os.path.join(IMAGESET_DIR,"_pepper_val.txt")

# save trainval.txt
# 末尾にリストの内容を追加する
with open(_trainval_images, "w", encoding="utf-8") as f:
  f.write("\n".join(trainval_images))

with open(_trainval_segmentation, "w", encoding="utf-8") as f:
  f.write("\n".join(trainval_segment))

with open(_pepper_trainval, "w", encoding="utf-8") as f:
  f.write("\n".join(pepper_trainval))

with open(_train_images, "w", encoding="utf-8") as f:
  f.write("\n".join(train_images))

with open(_train_segmentation, "w", encoding="utf-8") as f:
  f.write("\n".join(train_segment))

with open(_pepper_train, "w", encoding="utf-8") as f:
  f.write("\n".join(pepper_train))

with open(_val_images, "w", encoding="utf-8") as f:
  f.write("\n".join(val_images))

with open(_val_segmentation, "w", encoding="utf-8") as f:
  f.write("\n".join(val_segment))

with open(_pepper_val, "w", encoding="utf-8") as f:
  f.write("\n".join(pepper_val))
