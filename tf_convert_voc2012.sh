#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# Script to download and preprocess the PASCAL VOC 2012 dataset.
#
# Usage:
#   bash ./download_and_convert_voc2012.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_data.py
#     - build_voc2012_data.py
#     - download_and_convert_voc2012.sh
#     - remove_gt_colormap.py
#     + pascal_voc_seg
#       + VOCdevkit
#         + VOC2012
#           + JPEGImages
#           + SegmentationClass
#

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./pepper_pascal_voc_seg"
# WORK_DIR="./pascal_voc_seg"


cd "${CURRENT_DIR}"
# Root path for PASCAL VOC 2012 dataset.
PEPPER_PASCAL_ROOT="${WORK_DIR}/VOCdevkit/VOC2012"

# Remove the colormap in the ground truth annotations.
# colormap付きのセグメンテーション画像
SEG_FOLDER="${PEPPER_PASCAL_ROOT}/SegmentationClass"
# colormapなしのセグメンテーション画像
SEMANTIC_SEG_FOLDER="${PEPPER_PASCAL_ROOT}/SegmentationClassRaw"

echo "Removing the color map in ground truth annotations..."
# PIL → ndarrayでcolorをremoveできる
python3 ./remove_gt_colormap.py \
  --original_gt_folder="${SEG_FOLDER}" \
  --output_dir="${SEMANTIC_SEG_FOLDER}"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${PEPPER_PASCAL_ROOT}/JPEGImages"
LIST_FOLDER="${PEPPER_PASCAL_ROOT}/ImageSets/Segmentation"

echo "Converting PASCAL VOC 2012 dataset..."
python3 ./build_voc2012_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"
