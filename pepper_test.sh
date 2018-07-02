# copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local test on PASCAL VOC 2012 using MobileNet-v2.
# Users could also modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test_mobilenetv2.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
# Run model_test first to make sure the PYTHONPATH is correctly set.
# python3 "${WORK_DIR}"/model_test.py -v

# create TF data
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"
#sh tf_convert_voc2012.sh

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
PASCAL_FOLDER="pepper_pascal_voc_seg"
EXP_FOLDER="exp/train_on_trainval_set_mobilenetv2"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# モデルの指定
CKPT_NAME="deeplabv3_mnv2_pascal_train_aug"
PASCAL_DATASET="${WORK_DIR}/${DATASET_DIR}/${PASCAL_FOLDER}/tfrecord"

# 学習
# NUM_ITERATIONS=30000
# python3 "${WORK_DIR}"/train.py \
#   --logtostderr \
#   --train_logdir="${TRAIN_LOGDIR}" \
#   --log_steps=10 \
#   --save_interval_secs=1200 \
#   --save_summaries_secs=600 \
#   --save_summaries_images=False \
#   --training_number_of_steps="${NUM_ITERATIONS}" \
#   --train_batch_size=12 \
#   --train_crop_size=513 \
#   --train_crop_size=513 \
#   --upsample_logits=True \
#   --tf_initial_checkpoint="${INIT_FOLDER}/${CKPT_NAME}/model.ckpt-30000" \
#   --initialize_last_layer=False \
#   --last_layers_contain_logits_only=False \
#   --fine_tune_batch_norm=True \
#   --train_split="train" \
#   --model_variant="mobilenet_v2" \
#   --atrous_rates=1 \
#   --output_stride=16 \
#   --dataset="pepper" \
#   --dataset_dir="${PASCAL_DATASET}"


# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=75.34%.
python "${WORK_DIR}"/eval0.py \
  --logtostderr \
  --eval_logdir="${EVAL_LOGDIR}" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_batch_size=1 \
  --model_variant="mobilenet_v2" \
  --eval_crop_size=1025 \
  --eval_crop_size=2049 \
  --dataset='pepper' \
  --eval_split="val" \
  --dataset_dir="${PASCAL_DATASET}" \
  --max_number_of_evaluations=1

# Visualize the results.
# python3 "${WORK_DIR}"/vis.py \
#   --logtostderr \
#   --dataset='pepper' \
#   --vis_split="val" \
#   --model_variant="mobilenet_v2" \
#   --vis_crop_size=1025 \
#   --vis_crop_size=2049 \
#   --checkpoint_dir="${TRAIN_LOGDIR}" \
#   --vis_logdir="${VIS_LOGDIR}" \
#   --dataset_dir="${PASCAL_DATASET}" \
#   --colormap_type="pascal"
#   --max_number_of_iterations=1

# # Export the trained checkpoint.
# CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
# EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"
#
# python3 "${WORK_DIR}"/export_model.py \
#   --logtostderr \
#   --checkpoint_path="${CKPT_PATH}" \
#   --export_path="${EXPORT_PATH}" \
#   --model_variant="mobilenet_v2" \
#   --num_classes=22 \
#   --crop_size=513 \
#   --crop_size=513 \
#   --inference_scales=1.0
#
# # Run inference with the exported checkpoint.
# # Please refer to the provided deeplab_demo.ipynb for an example.
