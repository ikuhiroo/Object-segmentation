# 手順書
Classification :  predict the presence/absence  
Detection : predict the bounding boxes of each object    
Segmentation :  predict the class of the object containing that pixel or ‘background’  if the pixel does not belong to one of the twenty specified classes.  

## GPUに送るファイル  

## deeplab(tensorflow)をクローン  
`$ git clone https://github.com/tensorflow/models.git`  

## 環境構築， 学習済みモデルの取得  
`$cd models/research/deeplab `  
`$sh ./local_test.sh `    

## 自家製のデータセットで試す手順  
### ＜前処理＞   
作成中  
### ＜学習＞  
作成中
#### ・export the trained checkpoint.
#### ・export_model　
### ＜評価＞  
作成中  
### ＜Visualize＞
作成中


## 学習済みモデルのデータセット（pascal voc 2012）  
### ・[pascal voc devkit_doc](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf)  
### ・[pascal voc example](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html)  
### ・20 classes
pixel indices correspond to classes in alphabetical order  
(1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car , 8=cat, 9=chair, 10=cow, 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor)    

### ・[ROI（Region Of Interest）](http://www.orbit.bio/annotations/)  

## VOCdevkitの詳細情報  
### <詳細>  
### ●JPEGImages  
#### ・オリジナルのJPEG画像  

### ●Annotations  
#### ・画像内の情報（boxの座標やクラス名）  
#### ・`labelImg-master`でアノテーション付けを行う．  

### ●segmentation/SegmentationClass  
#### インデックスカラーによって着色された画像
### ●segmentation/SegmentationClassRaw  
#### ・ラベルデータ
##### SegmentationClassフォルダの画像の色を全部消して，エッジ検出のみをした画像を生成する．エッジの内部には各ラベルの色がグレースケールで書き込まれている．
#### ・` $ python check_convert.py `  
##### 用いられている輝度値の確認  （PASCAL_VOC）  
> {0: 361560627, 1: 3704393, 2: 1571148, 3: 4384132, 4: 2862913, 5: 3438963, 6: 8696374, 7: 7088203, 8: 12473466, 9: 4975284, 10: 5027769, 11: 6246382, 12: 9379340, 13: 4925676, 14: 5476081, 15: 24995476, 16: 2904902, 17: 4187268, 18: 7091464, 19: 7903243, 20: 4120989, 255: 28568409}) 

### ●segmentation/SegmentationObject  
#### ・SegmentationClassとの違い？

### ●ImageSets/Main  
#### ・AnnocationsとJPEGImagesのファイル名(拡張子を除く)のpair 
>ex. Main/train.txtにおける1行目（2008_000008）  
JPEGImages/2008_000008.jpg  
Annotations/2008_000008.xml  
#### ・クラス名ごとに，trainval.txt, train.txt, val.txtがある．  
どこで呼び出されているかが不明？
`{class_name}_trainval.txt`  
`{class_name}_train.txt`  
`{class_name}_val.txt`  
#### ・trainval.txt, train.txt, val.txt
どこで呼び出されているかが不明？
`trainval.txt`
`train.txt`
`val.txt`

### ●ImageSets/Segmentation  
#### ・trainval.txt, train.txt, val.txtを参考にしてtf-recodeにconvertされる  
```
tfrecodeディレクトリ内のファイルと対応

trainval-00000-of-00004.tfrecord
trainval-00001-of-00004.tfrecord
trainval-00002-of-00004.tfrecord
trainval-00003-of-00004.tfrecord
train-00000-of-00004.tfrecord
train-00001-of-00004.tfrecord
train-00002-of-00004.tfrecord
train-00003-of-00004.tfrecord
val-00000-of-00004.tfrecord
val-00001-of-00004.tfrecord
val-00002-of-00004.tfrecord
val-00003-of-00004.tfrecord
```

### ●ImageSets/Action  
どこで呼び出されているかが不明？
#### ・personクラスに対して，動作ごとにまとめる   

### ●ImageSets/Layout  
どこで呼び出されているかが不明？
#### ・1か2に振られる
>filename 1/2 

## 新規のクラスを追加する場合  
### ●pepper_test.shにおける設定
#### ・ working directoriesのSet up
`PASCAL_FOLDER="pepper_pascal_voc_seg"`
`EXP_FOLDER="exp/train_on_trainval_set_mobilenetv2"`
#### ・モデルの指定
`CKPT_NAME="deeplabv3_mnv2_pascal_train_aug"`

### ●trainvalの分割方法の設定し，ファイルを追加する  
 ` $python create_trainval.py `  
### ●用いられている輝度値の確認    
#### ・` $ python check_convert.py `  （PASCAL_VOC + PEPPER）  
>{0: 441731424, 1: 3704393, 2: 1571148, 3: 4384132, 4: 2862913, 5: 3438963, 6: 8696374, 7: 7088203, 8: 12473466, 9: 4975284, 10: 5027769, 11: 6246382, 12: 9379340, 13: 4925676, 14: 5476081, 15: 24995476, 16: 2904902, 17: 4187268, 18: 7091464, 19: 7903243, 20: 4120989, 21: 9762030, 255: 28568409})  
 
#### ・background（黒） → 0  
#### ・edge（白）→ 255  

### ●アノテーションツール（labelImg）  
#### ・boxの位置とクラス名  

### ●アノテーションツール（labelme）  
#### ・semantic segmentationを作成する．  
#### ・アノテーションツールの導入  
`$ pip install labelme ` 
#### ・アノテーションツールの使い方
 `$ labelme `  
→ jsonファイルの作成  
 ` $ python json_to_png.py   `
→ pngファイルの作成  

### ●trainvalの分割方法の設定し，ファイルに情報を追加する  
 ` $python create_trainval.py `  
>pascal  
images_train_val : 11540  
images_train : 5717  
images_val : 5823  
segment_train_val : 2913  
segment_train : 1464  
segment_val : 1449 

>pascal + pepper  
images_train_val : 11755  
images_train : 5889  
images_val : 5866  
segment_train_val : 3128  
segment_train : 1636  
segment_val : 1492

### ●segmentation_datasetファイルの変更  
#### ・_DATASETS_INFORMATIONの内容に追加  
##### cityscapes : _CITYSCAPES_INFORMATION  
##### pascal_voc_seg : _PASCAL_VOC_SEG_INFORMATION  
##### ade20k : _ADE20K_INFORMATION  

#### ・pepperに関するDatasetDescriptorの生成  
##### DatasetDescriptorの内容（base）  
##### ・splits_to_sizes  
##### Splits of the dataset into training, val, and test.    
##### ・num_classes    
##### Number of semantic classes.  
##### ・ignore_label
##### 255（edge）→クラスではなく境界なので除く

#### ex.  pepperの画像を加え，22クラスにした場合，
```_PEPPER_INFORMATION = DatasetDescriptor(
    splits_to_sizes={  
        'train': 1636,  
        'trainval': 3128,  
        'val': 1492,  
    },  
    num_classes=22,  
    ignore_label=255,  
)
```  
## create tf-recode
#### `sh tf_convert_voc2012.sh`
#### ・`WORK_DIR=（パス指定）`

## train
### ハイパーパラメータ設定（defaultはfine-tuning前の値）
#### ●Settings for logging.  
#### ・`train_logdir = (パス指定)`  
##### Where the checkpoint and logs are stored.  
##### default : (パス指定)
#### ・`log_steps = 10`  
##### Display logging information at every log_steps.  
##### default : 10
#### ・`save_interval_secs = 1200`  
##### How often, in seconds, we save the model to disk.  
##### default : 1200
#### ・`save_summaries_secs = 600`  
##### How often, in seconds, we compute the summaries.  
##### default : 600
#### ・`save_summaries_images = True`  
##### Save sample inputs, labels, and semantic predictions as images to summary.  
##### default : False
##### test1 : False

#### ●Settings ModelOptions.    
#### ・`model_variant="mobilenet_v2"` 
##### DeepLab model variant   
##### common.pyで設定
#### ・`training_number_of_steps = 30000`  
##### The number of steps used for training  
##### default : 30000
##### test1 : 10000
#### ・`train_batch_size = 4`  
##### The number of images in each batch during training.  
##### default : 4
##### test1 : 12
#### ・`fine_tune_batch_norm = True`  
##### Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3
##### When fine_tune_batch_norm=True, use at least batch size larger than 12  
##### Set to False and use small batch size to save GPU memory
##### otherwise use smaller batch.  
##### default : True
#### ・`upsample_logits = True`  
##### default :  True
##### Upsample logits during training.  
#### ・`tf_initial_checkpoint = (パス指定)`  
##### Settings for fine-tuning the network.
##### default : (パス指定)
#### ・`base_learning_rate = .0001`
##### The base learning rate for model training.
##### upsample_logits = True(when training on PASCAL augmented training set) → Use 0.007
###### tf_initial_checkpoint != None(When fine-tuning on PASCAL trainval set) → use learning rate=0.0001.
##### default : .0001
##### test1 :  .0001
#### ・`initialize_last_layer = False`  
##### Initialize last layer or not.  
##### Set to False if one does not want to re-use the trained classifier weights.
##### クラス数を増やしたため，最終層は学習済みモデルを使わない→False
##### default : True
##### test1 :  False
#### ・`last_layers_contain_logits_only = False`  
##### Only consider logits as last layers or not.  
##### default : False
##### test1 :  False
#### ・`train_crop_size=513 \ train_crop_size=513`
##### Image crop size [height, width] during training.
##### default : [513, 513]
##### test1 :  [513, 513]
#### ・`atrous_rates = None`
##### A list of atrous convolution rates for ASPP.
##### Atrous rates for atrous spatial pyramid pooling.
##### xception_65の場合，
##### output_stride = 8 → atrous_rates = [12, 24, 36]
##### output_stride = 16 → atrous_rates = [6, 12, 18]
##### mobilenet_v2の場合，
##### atrous_rates = None
##### default : None
##### test1 : 1（原因？）
##### one could use different atrous_rates/output_stride during training/evaluation.
#### ・`output_stride = 16`
##### The ratio of input to output spatial resolution.
##### default : 16
##### test1 : 16

#### ●Dataset settings.  
#### ・`dataset = pepper`  
##### Name of the segmentation dataset.  
#### ・`train_split = train`  
##### Which split of the dataset to be used for training  
##### default : trainval
#### ・`dataset_dir = (パス指定)`  
##### Where the dataset reside.  

## eval  
### ●ハイパーパラメータ設定（defaultはfinetuning前の値）
#### Settings ModelOptions.    
#### ・`logtostderr`  
#### ・`eval_logdir= (パス指定)`  
#### ・`checkpoint_dir= (パス指定)`   
#### ・`eval_batch_size=1`   
#### ・`model_variant="mobilenet_v2"`    
#### ・`eval_crop_size=800`   
##### Set eval_crop_size = output_stride * k + 1 for your dataset. 
##### The default value, 513, is set for PASCAL images whose largest image dimension is 512. 
##### default : 513 
#### ・`eval_crop_size=1200`  
##### default : 513
#### Dataset settings.  
#### ・`dataset='pepper'`   
#### ・`eval_split="val"`  
#### ・`dataset_dir=（パス指定）`    
#### ・`max_number_of_evaluations=1`    

### ●エラーメッセージ
#### Shape mismatch in tuple component 1. Expected [513,513,3], got [800,1200,3] .

### ●[tf.metric](https://electric-blue-industries.com/wp/machine-learnings/python-modules/python-modules-tensorflow/tf-metrics/)  を用いて評価

#### ・tf.metric.accuracy
##### 正解率, (過去の合計正答数total) / (データ数)
#### ・mean_per_class_accuracy
##### クラスごとの精度の平均  
#### ・precision
##### 適合率    
#### ・false_negatives
##### 偽陰性の総数    
#### ・miou_1.0
##### boxに対して, 目的となる領域(ground truth box)がどれだけ含まれているか

### ●初期の学習済みモデルを用いて，１０回学習させた場合  
#### ・結果1（`python eval.py`）
```
accuracy[0.943350315]  
mean_per_class_accuracy[0.847709656]  
precision[0.922471046]  
false_negatives[0.687992334]  
miou_1.0[0.75342977]  
```
#### ・結果2（`python eval0.py`）
```
background   0_accuracy: 86.5819690551117 % 
aeroplane    1_accuracy: 93.99845722278008 % 
bicycle      2_accuracy: 81.85716486131588 % 
bird         3_accuracy: 89.31383046180657 % 
boat         4_accuracy: 89.2741476044005 % 
bottle       5_accuracy: 67.78285804004202 % 
bus          6_accuracy: 92.48821158813442 % 
car          7_accuracy: 81.50931779330695 % 
cat          8_accuracy: 94.63598087766903 % 
chair        9_accuracy: 56.628017201823795 % 
cow          10_accuracy: 93.8124551416695 % 
diningtable  11_accuracy: 60.31791656604843 % 
dog          12_accuracy: 93.8508213015269 % 
horse        13_accuracy: 93.23947809726272 % 
motorbike    14_accuracy: 92.92242241704558 % 
person       15_accuracy: 83.55709018384393 % 
potted plant 16_accuracy: 61.9201044111386 % 
sheep        17_accuracy: 91.03804536000922 % 
sofa         18_accuracy: 68.10712464936924 % 
train        19_accuracy: 96.32077194416111 % 
tv/monitor   20_accuracy: 79.72798036113859 % 
```
### ●新しい画像を使って，学習させた結果
#### --train_crop_size=1025 \--train_crop_size=2049 \
#### ・結果1（`python eval.py`）
```
accuracy[0.887568057]  
mean_per_class_accuracy[0.49432373]     
precision[0.961355925]  
false_negatives[0.32001844]   
miou_1.0[0.441790938]  
```
#### ・結果2（`python eval0.py`）
```
background   0_accuracy: 96.84267431853029 % 
aeroplane    1_accuracy: 57.357618018805816 % 
bicycle      2_accuracy: 0.0 % 
bird         3_accuracy: 49.55682088565027 % 
boat         4_accuracy: 16.517761537994804 % 
bottle       5_accuracy: 0.08028326685488951 % 
bus          6_accuracy: 72.51021481737355 % 
car          7_accuracy: 45.66972298631186 % 
cat          8_accuracy: 75.63807704904475 % 
chair        9_accuracy: 0.13013601331749589 % 
cow          10_accuracy: 61.40326484451494 %
diningtable  11_accuracy: 9.461895429316897 % 
dog          12_accuracy: 76.36539074135965 % 
horse        13_accuracy: 68.5181200573232 % 
motorbike    14_accuracy: 57.87662859525158 % 
person       15_accuracy: 71.1393828885102 % 
potted plant 16_accuracy: 0.0 % 
sheep        17_accuracy: 46.77391369287723 % 
sofa         18_accuracy: 56.46067461595401 % 
train        19_accuracy: 65.60469802975989 % 
tv/monitor   20_accuracy: 40.424908601558336 % 
Pepper       21_accuracy: 87.54083190904221 % 
```
#### ・考察
##### ・pepperの画像を加えると，PASCALのクラスに対する認識率が下がる→pepper認識器
##### ・クラスごとの精度の平均が下がっている．  
##### ・boxの精度も下がっている．  
##### ・アノテーション付けがよくないのか？  
##### ・crop_sizeが異なるのがダメなのか？
##### PASCAL VOCに関して
```
・max_weight = 500
・max_height = 500
・widhtの大きさ別の辞書
{'500<': 0, '<501': 13391, '<401': 3516, '<301': 203, '<201': 14, '<101': 0}
・heightの大きさ別の辞書
{'500<': 0, '<501': 4199, '<401': 12285, '<301': 574, '<201': 64, '<101': 2}
```
##### 新規データセットに関して
```
・max_weight = 5760
・max_height = 2998
・widhtの大きさ別の辞書
{'500<': 97, '<501': 13415, '<401': 3543, '<301': 253, '<201': 31, '<101': 1}
・heightの大きさ別の辞書
{'500<': 83, '<501': 4230, '<401': 12324, '<301': 607, '<201': 94, '<101': 2}
```
##### →画像サイズを500以下にすることでcrop_sizeを513にできる
```
・max_weight = 500
・max_height = 500
・widhtの大きさ別の辞書
{'500<': 0, '<501': 13512, '<401': 3543, '<301': 253, '<201': 31, '<101': 1}
・heightの大きさ別の辞書
{'500<': 0, '<501': 4313, '<401': 12324, '<301': 607, '<201': 94, '<101': 2}
```

## Visualize  
### ハイパーパラメータ設定  
#### ・`logtostderr`
#### ・`dataset='pepper'`  
#### ・`vis_split="val"`
#### ・`model_variant="mobilenet_v2"`
#### ・`vis_crop_size=1025`
##### default : 513
#### ・`vis_crop_size=2049`
##### default : 513
#### ・`checkpoint_dir=（パス指定）`
#### ・`vis_logdir=（パス指定）`
#### ・`dataset_dir=（パス指定）`
#### ・`colormap_type="pascal"`
#### ・`max_number_of_iterations=1`

## export_model  
### ハイパーパラメータ設定
#### ・`--logtostderr`
#### ・`checkpoint_path=（パス指定）`
#### ・`export_path=（パス指定）"`
#### ・`model_variant="mobilenet_v2"`
#### ・`num_classes=22`  
#### ・`crop_size=513`
#### ・`crop_size=513`
#### ・`inference_scales=1.0`

> Written with [StackEdit](https://stackedit.io/).
