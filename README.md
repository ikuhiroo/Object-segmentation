# Object-segmentation

手順書

●deeplab(tensorflow)をクローン  
git clone https://github.com/tensorflow/models.git  
●データセット，学習済みモデルの取得，一部学習  
cd models/research/deeplab  
sh ./local_test.sh  

●自家製のデータセットを食わせる手順    
＜前処理＞    
・アノテーション付け, セグメンテーション付け	  
・label_map.pbtxtの作成	
・train_with_masks.recordとval_with_masks.recordの作成	
同時に, val.txtの作成	  
・train_val.txtの作成	  
・configファイルの作成	  
＜学習＞	  
＜評価＞	  
・pbファイルの作成	  
・jupyter notebookで評価	

●データセット（pascal voc 2012）	
<参考>http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf	
19 classes.	
• person	
• bird, cat, cow, dog, horse, sheep • aeroplane, bicycle, boat, bus, car, motorbike, train	
• bottle, chair, dining table, potted plant, sofa, tv/monitor	

The train/val data has 11,540 images containing ? ROI annotated objects and ? segmentations.	
ROI（Region Of Interest）・・・http://www.orbit.bio/annotations/	

<詳細>	
●JPEGImages(17125)
（説明）
オリジナルのJPEG画像

●Annotations(17125)
（説明）
画像内の情報（boxとそのクラス名）

●segmentation/SegmentationClass（2913）
（説明）
VOCのSegmentationClassに存在する画像はインデックスカラーによって着色されている．
pngで画像を読み込めばそのままクラスを表すarrayになる．

21クラス(index color, 背景を含めている)
セグメンテーションされたpng画像が格納されている。
・ImageReaderでデコードする．
tf.image.decode_png
output : （縦、横、チャンネル）
channels : 1

（作成方法）
●今あるデータセットの輝度値を取得．
クラス数 : 20クラス（背景を含む）
defaultdict(<function <lambda> at 0x100c70e18>,
{0: 361560627, 220: 28568409, 38: 3704393, 147: 24995476, 52: 7559952, 150: 7903243, 14: 2862913, 57: 4975284, 33: 9379340, 113: 4384132, 75: 5758416, 132: 6246382, 72: 4925676, 108: 5476081, 112: 7091464, 94: 5027769, 128: 7088203, 19: 12473466, 89: 8696374, 37: 2904902})

●使える輝度を保管．
背景 → 0
pepperのface → 40
●pepper画像をアノテーションツールを用いて，semantic segmentationを作成する．
（アノテーションツールの使い方）
pylabelme-master
$ labelme
→ jsonファイルが作成
→ pngファイルの作成
$ labelme_json_to_dataset jsonファイル -o フォルダ名
→ img.png, info.yaml, label_names.txt, label_viz.png, label.png

・ソースの変更
json → label.png → segmentation.png
輝度値を振り分ける方法

●segmentation/SegmentationClassRaw（2913）
（説明）
グレースケールの画像
RGB情報をグレースケールに変換する処理
Folder containing semantic segmentation annotations
remove_gt_colormapでmodified ground truth annotations

●segmentation/SegmentationObject（2913）
（説明）

●ImageSets/Main
（構成）
/trainval.txt(11540) + pepperの情報
/train.txt(5717) + pepperの情報
/val.txt(5823) + pepperの情報
（説明）
AnnocationsとJPEGImagesディレクトリのファイル名(拡張子を除く)を示していて、
画像ファイルと、その正解データの組を表現しています。
ex. Main/train.txtにおける1行目（2008_000008）
JPEGImages/2008_000008.jpg
Annotations/2008_000008.xml
（新規の画像）
python create_trainval.py
trainvalの分割方法の設定
形式に従って，ファイルに追加する

分類先に関して1 / -1の値,
●ImageSets/Segmentation
（構成）
/trainval(2913) + pepper
/train(1464) + pepper
/val(1449) + pepper
（説明）
outputs_to_num_classes['semantic']=21
学習/評価に用いる画像ファイル名を作成
（新規の画像）
trainvalの分割方法の設定
形式に従って，ファイルに追加する

●ImageSets/Action
（構成）
/trainval(4588)
/train(2296)
/val(2292)
（説明）
personクラスに対する動作ごと

●ImageSets/Layout
（構成）
/trainval(850)
/train(425)
/val(425)

●tf-recordの作成
（説明）
./VOCdevkit/VOC2012/ImageSets/Segmentation
を参照して，train_val.txt/train.txt/val.txtをまとめる．
続いて，順番にtf_recordを作成する．４つに分割．
・JPEGImages（3channel）
・SegmentationClass（1channel）
を参照して，image_filenameとseg_filenameを作成する．
この際，height == seg_height，width == seg_width．

●segmentation_dataset.py
（説明）
tf_recordをモデルに入れるに変換．

_DATASETS_INFORMATIONの内容に追加
・cityscapes : _CITYSCAPES_INFORMATION
・pascal_voc_seg : _PASCAL_VOC_SEG_INFORMATION
・ade20k : _ADE20K_INFORMATION
・pepperに関するDatasetDescriptorの生成

DatasetDescriptorの内容（base）
・splits_to_sizes（Splits of the dataset into training, val, and test）
・num_classes（Number of semantic classes.）
・ignore_label（Ignore label value.）背景
ex.
_PASCAL_VOC_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 1464,
        'trainval': 2913,
        'val': 1449,
    },
    num_classes=21,
    ignore_label=255,
)

get_dataset(dataset, train_split, dataset_dir)
Gets an instance of slim Dataset
dataset(=dataset_name) : 作成したtf_record
train_split(=split_name) : trainval
dataset_dir :

●segmentation_datasetの変更
"""pepper情報追加"""
_PEPPER_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 1636,
        'trainval': 3128,
        'val': 1492,
    },
    num_classes=22,
    ignore_label=255,
)

_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
    'pepper': _PEPPER_INFORMATION
}


●学習済みモデル
init_models/deeplabv3_mnv2_pascal_train_aug

●学習
WORK_DIR : /Users/1-10robotics/Desktop/google/models-master/research
$ python deeplab/train.py
--train_logdir=/Users/1-10robotics/Desktop/google/models-master/research/deeplab/train --tf_initial_checkpoint=/Users/1-10robotics/Desktop/google/models-master/research/deeplab/deeplabv3_mnv2_pascal_train_aug/model.ckpt-30000 --dataset_dir=/Users/1-10robotics/Desktop/google/models-master/research/deeplab/datasets/pascal_voc_seg/tfrecord

●評価
・exp/trainのmodelをexportへ移動
・frozen graphの作成
WORK_DIR : /Users/1-10robotics/Desktop/google/models-master/research
$ python deeplab/export_model.py
--logtostderr --checkpoint_path=/Users/1-10robotics/Desktop/google/models-master/research/deeplab/datasets/pascal_voc_seg/exp/export/model.ckpt-54 --export_path=/Users/1-10robotics/Desktop/google/models-master/research/deeplab/datasets/pascal_voc_seg/exp/export/frozen_inference_graph.pb

・学習済みモデル（.index, .data, flozen）
$ tar -zcvf ..export.tar.gz ..export

・jupyter notebookの実行
# five main task
Classification :  predict the presence/absence
Detection : predict the bounding boxes of each object
Segmentation :
predict the class of the object containing that pixel
or ‘background’  if the pixel does not belong to one of the twenty specified classes.
