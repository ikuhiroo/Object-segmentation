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

"""Removes the color map from segmentation annotations.

Removes the color map from the ground truth segmentation annotations and save
the results to output_dir.
"""
import glob
import os.path
import sys
import numpy as np
from collections import defaultdict

from PIL import Image

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('original_gt_folder',
                           './VOCdevkit/VOC2012/SegmentationClass',
                           'Original ground truth annotations.')

tf.app.flags.DEFINE_string('segmentation_format', 'png', 'Segmentation format.')

tf.app.flags.DEFINE_string('output_dir',
                           '/Users/1-10robotics/Desktop/files/output',
                           'folder to save modified ground truth annotations.')


def _remove_colormap(filename):
  """Removes the color map from the annotation.

  Args:
    filename: Ground truth annotation filename.
    PIL → np

  Returns:
    Annotation without color map.
  """
  return np.array(Image.open(filename))


def _save_annotation(annotation, filename):
  """Saves the annotation as png file.

  Args:
    annotation: Segmentation annotation.
    filename: Output filename.
  """
  pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
  print('pil_image.shape : {}'.format(pil_image.size))
  print('pil_image.dtype : {}'.format(pil_image.format))
  with tf.gfile.Open(filename, mode='w') as f:
    pil_image.save(f, 'PNG')


def main(unused_argv):
  # 引数設定
  argv = sys.argv
  argc = len(argv)
  if (argc < 2):
    print('Usage: python %s csvfile' %argv[0])
    quit()
  # Create the output directory if not exists.
  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  # annotations = glob.glob(os.path.join(FLAGS.original_gt_folder, '*.' + FLAGS.segmentation_format))
  annotation = argv[1]
  print(annotation)

  raw_annotation = _remove_colormap(annotation)
  print('raw_annotation : {}'.format(raw_annotation.shape))
  print('raw_annotation : {}'.format(raw_annotation.dtype))
  print('raw_annotation : {}'.format(raw_annotation))

  filename = os.path.basename(annotation)[:-4]
  _save_annotation(raw_annotation, os.path.join(FLAGS.output_dir, filename + '.' + FLAGS.segmentation_format))

  color_value = defaultdict(lambda: 0)
  for p in range(len(raw_annotation)):
      for q in range(len(raw_annotation[p])):
              color_value[int(raw_annotation[p][q])] += 1
  print(color_value)
  print('Saved to: %s' % FLAGS.output_dir)

if __name__ == '__main__':
  tf.app.run()
