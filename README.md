### 手順書
Classification :  predict the presence/absence  
Detection : predict the bounding boxes of each object    
Segmentation :  predict the class of the object containing that pixel or ‘background’  if the pixel does not belong to one of the twenty specified classes.  
```
```
#### GPUに送るファイル  

#### deeplab(tensorflow)をクローン  
 `$ git clone https://github.com/tensorflow/models.git`  
 #### 環境構築， 学習済みモデルの取得  
 `$cd models/research/deeplab `  
 `$sh ./local_test.sh `    
```
```
***
#### 自家製のデータセットを食わせる手順  
＜前処理＞   
作成中  
＜学習＞  
作成中　　
＜評価＞  
作成中  
```
```
***
#### 学習済みモデルのデータセット（pascal voc 2012）  
[pascal voc devkit_doc](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf)  
20 classes.  
• person  
• bird, cat, cow, dog, horse, sheep • aeroplane, bicycle, boat, bus, car, motorbike, train  
• bottle, chair, dining table, potted plant, sofa, tv/monitor  
```
```
[ROI（Region Of Interest](http://www.orbit.bio/annotations/)  
```
```
#### VOCdevkitの詳細情報  
<詳細>  
●JPEGImages  
（説明）  
オリジナルのJPEG画像  
```
```
●Annotations  
（説明）  
画像内の情報（boxの座標やクラス名）  
`labelImg-master`でアノテーション付けを行う．  
```
```
●segmentation/SegmentationClass  
（説明）  
VOCのSegmentationClassにある画像はインデックスカラーによって着色されている．   
```
```
●segmentation/SegmentationClassRaw  
（説明）  
クラスのidが輝度値として使用されている．  
` $ python check_convert.py `  
用いられている輝度値の確認  
（PASCAL_VOC）  
> {0: 361560627, 1: 3704393, 2: 1571148, 3: 4384132, 4: 2862913, 5: 3438963, 6: 8696374, 7: 7088203, 8: 12473466, 9: 4975284, 10: 5027769, 11: 6246382, 12: 9379340, 13: 4925676, 14: 5476081, 15: 24995476, 16: 2904902, 17: 4187268, 18: 7091464, 19: 7903243, 20: 4120989, 255: 28568409}) 

```
```
●segmentation/SegmentationObject  
（説明）
●ImageSets/Main  
（説明）  
AnnocationsとJPEGImagesディレクトリのファイル名(拡張子を除く)を示し、  
画像ファイルと、その正解データの組を表現している．  
>ex. Main/train.txtにおける1行目（2008_000008）  
JPEGImages/2008_000008.jpg  
Annotations/2008_000008.xml  

```
```
（構成）  
クラス名ごとに，trainval.txt, train.txt, val.txtがある．  
`{class_name}_trainval.txt`  
`{class_name}_train.txt`  
`{class_name}_val.txt`  
```
```
●ImageSets/Segmentation  
（説明）  
trainval.txt, train.txt, val.txtが，tf-recordにconvertされる  
（構成）   
`filename`
```
```
●ImageSets/Action  
（説明）  
personクラスに対して存在し，動作ごとにファイルがある   
●ImageSets/Layout  
```
```
***
#### 新規のクラスを追加する場合  
●trainvalの分割方法の設定し，ファイルを追加する  
 ` $python create_trainval.py `  
●用いられている輝度値の確認    
 ` $ python check_convert.py `  
>（PASCAL_VOC）  
{0: 361560627, 1: 3704393, 2: 1571148, 3: 4384132, 4: 2862913, 5: 3438963, 6: 8696374, 7: 7088203, 8: 12473466, 9: 4975284, 10: 5027769, 11: 6246382, 12: 9379340, 13: 4925676, 14: 5476081, 15: 24995476, 16: 2904902, 17: 4187268, 18: 7091464, 19: 7903243, 20: 4120989, 255: 28568409})  

```
```
>（PASCAL_VOC + PEPPER）  
{0: 441731424, 1: 3704393, 2: 1571148, 3: 4384132, 4: 2862913, 5: 3438963, 6: 8696374, 7: 7088203, 8: 12473466, 9: 4975284, 10: 5027769, 11: 6246382, 12: 9379340, 13: 4925676, 14: 5476081, 15: 24995476, 16: 2904902, 17: 4187268, 18: 7091464, 19: 7903243, 20: 4120989, 21: 9762030, 255: 28568409})  
 
背景（黒） → 0  
輪郭（白）→ 255  
```
```
●アノテーションツール（labelImg）  
（説明）  
boxの位置とクラス名  
●アノテーションツール（labelme）  
（説明）  
semantic segmentationを作成する．  
（アノテーションツールの導入）  
`$ pip install labelme ` 
（アノテーションツールの使い方）  
 `$ labelme `  
→ jsonファイルの作成  
 ` $ python json_to_png.py   `
→ pngファイルの作成  
●trainvalの分割方法の設定し，ファイルに情報を追加する  
 ` $python create_trainval.py `  
>pascal  
images_train_val : 11540  
images_train : 5717  
images_val : 5823  
segment_train_val : 2913  
segment_train : 1464  
segment_val : 1449 

```
```
>pascal + pepper  
images_train_val : 11755  
images_train : 5889  
images_val : 5866  
segment_train_val : 3128  
segment_train : 1636  
segment_val : 1492

●segmentation_datasetファイルの変更  
（説明）  
_DATASETS_INFORMATIONの内容に追加  
・cityscapes : _CITYSCAPES_INFORMATION  
・pascal_voc_seg : _PASCAL_VOC_SEG_INFORMATION  
・ade20k : _ADE20K_INFORMATION  
・pepperに関するDatasetDescriptorの生成  
```
```
DatasetDescriptorの内容（base）  
・splits_to_sizes  
Splits of the dataset into training, val, and test.    
・num_classes    
Number of semantic classes.  
・ignore_label  
Ignore label value.  
```
```
ex.  
`_PEPPER_INFORMATION = DatasetDescriptor(  
    splits_to_sizes={  
        'train': 1636,  
        'trainval': 3128,  
        'val': 1492,  
    },  
    num_classes=22,  
    ignore_label=255,  
)  `   
```
```
***
### train.pyのハイパーパラメータについて  
●Settings for logging.  
`train_logdir = None`  
Where the checkpoint and logs are stored.  
`log_steps = 10`  
Display logging information at every log_steps.  
`save_interval_secs = 1200`  
How often, in seconds, we save the model to disk.  
`save_summaries_secs = 600`  
How often, in seconds, we compute the summaries.  
`save_summaries_images = false`  
Save sample inputs, labels, and semantic predictions as images to summary.  
```
```
●Settings for training strategy.    
`training_number_of_steps = 30000`  
The number of steps used for training  
`train_batch_size = 8`  
The number of images in each batch during training.  
`upsample_logits = True`  
Upsample logits during training.  
```
```
●Settings for fine-tuning the network.     
`tf_initial_checkpoint = None`  
TensorFlow checkpoint for initialization.  
`initialize_last_layer = true`  
Initialize last layer or not.  
Set to False if one does not want to re-use the trained classifier weights.  
`last_layers_contain_logits_only = false`  
Only consider logits as last layers or not.  
`fine_tune_batch_norm = true`  
When fine_tune_batch_norm=True, use at least batch size larger than 12  
otherwise use smaller batch.  
```
```
●Dataset settings.  
`dataset = pepper`  
Name of the segmentation dataset.  
`train_split = train`  
Which split of the dataset to be used for training  
`dataset_dir = None`  
Where the dataset reside.  
```
```
●train  
```
```
***
●eval  
・Shape mismatch in tuple component 1. Expected [513,513,3], got [800,1200,3]  
Set eval_crop_size = output_stride * k + 1 for your dataset  
pepperのサイズとPASCAL_VOCのcrop_sizeの違い？  