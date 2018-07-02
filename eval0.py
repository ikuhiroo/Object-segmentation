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
"""Evaluation script for the DeepLab model.

See model.py for more details and usage.
# tf.contrib.training.StopAfterNEvalsHook
# hooksはepoch毎に追加で行いたいOperationやepochそのものの値を設定
# all_hooks = [StopAfterNEvalsHook(num_evals)]
# Run hook
# 調べたいことはopsにして更新
# monitored_session.MonitoredSessionでsessionを立ち上げる

"""

import six
import math
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
# 新たに追加したモジュール
from collections import defaultdict
import numpy as np
import operator
from sklearn.metrics import confusion_matrix
from tensorflow.python.summary import summary
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import monitored_session
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as tf_saver
import time
from tensorflow.python.training import training_util
from tensorflow.python.training import evaluation

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('eval_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for evaluating the model.

flags.DEFINE_integer('eval_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_multi_integer('eval_crop_size', [513, 513],
                           'Image crop size [height, width] for evaluation.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

# Dataset settings.

flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_integer('max_number_of_evaluations', 0,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')

# class_id
id_dict = {0 : 'edge', 1 : "aeroplane", 2 : "bicycle", 3 : "bird", 4 : "boat", 5 : "bottle", 6 : "bus", 7 : "car" , 8 : "cat", 9 : "chair", 10 : "cow", 11  :"diningtable", 12 : "dog", 13 : "horse", 14 : "motorbike", 15 : "person", 16 : "potted plant", 17 : "sheep", 18 : "sofa", 19 : "train", 20 : "tv/monitor", 21 : "pepper"}

class _StopAfterNEvalsHook(session_run_hook.SessionRunHook):
  """Run hook used by the evaluation routines to run the `eval_ops` N times."""

  def __init__(self, num_evals, log_progress=True):
    """Constructs the run hook.
    Args:
      num_evals: The number of evaluations to run for. if set to None, will
        iterate the dataset until all inputs are exhausted.
      log_progress: Whether to log evaluation progress, defaults to True.
    """
    # The number of evals to run for.
    self._num_evals = num_evals
    self._evals_completed = None
    self._log_progress = log_progress
    # Reduce logging frequency if there are 20 or more evaluations.
    self._log_frequency = (1 if (num_evals is None or num_evals < 20)
                           else math.floor(num_evals / 10.))

  def _set_evals_completed_tensor(self, updated_eval_step):
    self._evals_completed = updated_eval_step

  def before_run(self, run_context):
    # Called before each call to run().
    return session_run_hook.SessionRunArgs({
        'evals_completed': self._evals_completed
    })

  def after_run(self, run_context, run_values):
    # Called after each call to run().
    evals_completed = run_values.results['evals_completed']
    if self._log_progress:
      if self._num_evals is None:
        logging.info('Evaluation [%d]', evals_completed)
      else:
        if ((evals_completed % self._log_frequency) == 0 or
            (self._num_evals == evals_completed)):
          logging.info('Evaluation [%d/%d]', evals_completed, self._num_evals)
    if self._num_evals is not None and evals_completed >= self._num_evals:
      run_context.request_stop()

class SummaryAtEndHook(session_run_hook.SessionRunHook):
  """A run hook that saves a summary with the results of evaluation."""

  def __init__(self,
               log_dir=None,
               summary_writer=None,
               summary_op=None,
               feed_dict=None):
    """Constructs the Summary Hook.
    Args:
      log_dir: The directory where the summary events are saved to.  Used only
        when `summary_writer` is not specified.
      summary_writer: A `tf.summary.FileWriter` to write summary events with.
      summary_op: The summary op to run. If left as `None`, then all summaries
        in the tf.GraphKeys.SUMMARIES collection are used.
      feed_dict: An optional feed dictionary to use when evaluating the
        summaries.
    Raises:
      ValueError: If both `log_dir` and `summary_writer` are `None`.
    """
    self._summary_op = summary_op
    self._replace_summary_op = summary_op is None
    self._feed_dict = feed_dict
    self._summary_writer = summary_writer
    self._log_dir = log_dir
    if self._log_dir is None and self._summary_writer is None:
      raise ValueError('One of log_dir or summary_writer should be used.')

  def begin(self):
    if self._replace_summary_op:
      self._summary_op = summary.merge_all()
    self._global_step = training_util.get_or_create_global_step()

  def after_create_session(self, session, coord):
    # Called when new TensorFlow session is created.
    if self._summary_writer is None and self._log_dir:
      self._summary_writer = summary.FileWriterCache.get(self._log_dir)

  def end(self, session):
    global_step = training_util.global_step(session, self._global_step)
    summary_str = session.run(self._summary_op, self._feed_dict)
    if self._summary_writer:
      self._summary_writer.add_summary(summary_str, global_step)
      self._summary_writer.flush()

_USE_DEFAULT = 0
StopAfterNEvalsHook = _StopAfterNEvalsHook
get_or_create_eval_step = evaluation._get_or_create_eval_step

def evaluate_repeatedly(checkpoint_dir,
                        master='',
                        scaffold=None,
                        eval_ops=None,
                        feed_dict=None,
                        final_ops=None,
                        final_ops_feed_dict=None,
                        eval_interval_secs=60,
                        hooks=None,
                        config=None,
                        max_number_of_evaluations=None,
                        timeout=None,
                        timeout_fn=None):
  """Repeatedly searches for a checkpoint in `checkpoint_dir` and evaluates it.
  During a single evaluation, the `eval_ops` is run until the session is
  interrupted or requested to finish. This is typically requested via a
  `tf.contrib.training.StopAfterNEvalsHook` which results in `eval_ops` running
  the requested number of times.
  Optionally, a user can pass in `final_ops`, a single `Tensor`, a list of
  `Tensors` or a dictionary from names to `Tensors`. The `final_ops` is
  evaluated a single time after `eval_ops` has finished running and the fetched
  values of `final_ops` are returned. If `final_ops` is left as `None`, then
  `None` is returned.
  One may also consider using a `tf.contrib.training.SummaryAtEndHook` to record
  summaries after the `eval_ops` have run. If `eval_ops` is `None`, the
  summaries run immediately after the model checkpoint has been restored.
  Note that `evaluate_once` creates a local variable used to track the number of
  evaluations run via `tf.contrib.training.get_or_create_eval_step`.
  Consequently, if a custom local init op is provided via a `scaffold`, the
  caller should ensure that the local init op also initializes the eval step.
  Args:
    checkpoint_dir: The directory where checkpoints are stored.
    master: The address of the TensorFlow master.
    scaffold: An tf.train.Scaffold instance for initializing variables and
      restoring variables. Note that `scaffold.init_fn` is used by the function
      to restore the checkpoint. If you supply a custom init_fn, then it must
      also take care of restoring the model from its checkpoint.
    eval_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names
      to `Tensors`, which is run until the session is requested to stop,
      commonly done by a `tf.contrib.training.StopAfterNEvalsHook`.
    feed_dict: The feed dictionary to use when executing the `eval_ops`.
    final_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names
      to `Tensors`.
    final_ops_feed_dict: A feed dictionary to use when evaluating `final_ops`.
    eval_interval_secs: The minimum number of seconds between evaluations.
    hooks: List of `tf.train.SessionRunHook` callbacks which are run inside the
      evaluation loop.
    config: An instance of `tf.ConfigProto` that will be used to
      configure the `Session`. If left as `None`, the default will be used.
    max_number_of_evaluations: The maximum times to run the evaluation. If left
      as `None`, then evaluation runs indefinitely.
    timeout: The maximum amount of time to wait between checkpoints. If left as
      `None`, then the process will wait indefinitely.
    timeout_fn: Optional function to call after a timeout.  If the function
      returns True, then it means that no new checkpoints will be generated and
      the iterator will exit.  The function is called with no arguments.
  Returns:
    The fetched values of `final_ops` or `None` if `final_ops` is `None`.
  """
  eval_step = get_or_create_eval_step()

  # Prepare the run hooks.
  hooks = hooks or []
  if eval_ops is not None:
    update_eval_step = state_ops.assign_add(eval_step, 1)

    for h in hooks:
      if isinstance(h, StopAfterNEvalsHook):
        h._set_evals_completed_tensor(update_eval_step)  # pylint: disable=protected-access

    if isinstance(eval_ops, dict):
      eval_ops['update_eval_step'] = update_eval_step
    elif isinstance(eval_ops, (tuple, list)):
      eval_ops = list(eval_ops) + [update_eval_step]
    else:
      eval_ops = [eval_ops, update_eval_step]

  final_ops_hook = basic_session_run_hooks.FinalOpsHook(final_ops,
                                                        final_ops_feed_dict)
  hooks.append(final_ops_hook)

  num_evaluations = 0
  for checkpoint_path in checkpoints_iterator(
      checkpoint_dir,
      min_interval_secs=eval_interval_secs,
      timeout=timeout,
      timeout_fn=timeout_fn):

    session_creator = monitored_session.ChiefSessionCreator(
        scaffold=scaffold,
        checkpoint_filename_with_path=checkpoint_path,
        master=master,
        config=config)

    # sessionの開始
    with monitored_session.MonitoredSession(session_creator=session_creator, hooks=hooks) as session:
      logging.info('Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
      if eval_ops is not None:
        class_accuracy = defaultdict(list) #class : accuracy
        while not session.should_stop():
          # sessionをrunする
          _labels, _predictions, _cnt = session.run(eval_ops, feed_dict)

          print('eval_{}'.format(_cnt))

          # リストの要素ごとに比較をする
          result = {} # (true : pred) : cnt
          true_cnt = {} # true : cnt
          result_all = {} # id : cnt

          true_len = 0
          for (true, pred) in zip(_labels, _predictions):
            if true == pred:
              true_len += 1
              try:
                true_cnt[true] += 1
              except:
                true_cnt[true] = 1
            else:
              try:
                result[(true, pred)] += 1
              except:
                result[(true, pred)] = 1
            try:
              result_all[true] += 1
            except:
              result_all[true] = 1

          result = sorted(result.items(), key=lambda x:x[1], reverse=True)

          # matrix = confusion_matrix(_labels, _predictions)
          misclassificated_pair_1 = result[0][0]
          misclassificated_rate_1 = (result[0][1]/sum(dict(result).values()))*100

          misclassificated_pair_2 = result[1][0]
          misclassificated_rate_2 = (result[1][1]/sum(dict(result).values()))*100

          misclassificated_pair_3 = result[2][0]
          misclassificated_rate_3 = (result[2][1]/sum(dict(result).values()))*100

          print('{} : {} %'.format(misclassificated_pair_1, misclassificated_rate_1))
          print('{} : {} %'.format(misclassificated_pair_2, misclassificated_rate_2))
          print('{} : {} %'.format(misclassificated_pair_3, misclassificated_rate_3))

          # mean_per_class_accuracy : クラスごとの精度の平均を計算します
          for i in range(len(result_all.keys())):
            key = list(result_all.keys())[i]
            try:
              accuracy = (true_cnt[key]/result_all[key])*100
              print('{}_accuracy: {} %'.format(key, accuracy))
            except:
              accuracy = 0
              print('{}_accuracy: {} %'.format(key, accuracy))

            class_accuracy[key].append(accuracy)

          # accuracy : 予測がラベルに一致する頻度を計算
          accuracy = (true_len / len(_labels))*100
          print("accuracy : {} %".format(accuracy))

          print(' ')

      # クラス平均
      for i in range(len(list(class_accuracy.keys()))):
        key = list(class_accuracy.keys())[i]
        ave_accuracy = sum(class_accuracy[key])/len(class_accuracy[key])
        print('{}_accuracy: {} %'.format(key, ave_accuracy))

      logging.info('Finished evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
    num_evaluations += 1

    if (max_number_of_evaluations is not None and
        num_evaluations >= max_number_of_evaluations):
      return final_ops_hook.final_ops_values

  return final_ops_hook.final_ops_values

# eval_1
# (0, 12) : 29.17293496134066 %
# accuracy : 97.27810115582855
# a = {}
# a[(1, 2)] = 1
# a[(2, 3)] = 2
# a[(1, 4)] = 3
# a[(4, 5)] = 4
# a[(1, 6)] = 5
# a[(6, 7)] = 6
# # b = sorted(a.items(), key=lambda x:x[1], reverse=True)
# for i in range(len(a.values())):
#   print(list(a.items())[i][0][0])
# sum(dict(b).values())
# b[0][0]

def evaluation_loop(master,
                    checkpoint_dir,
                    logdir,
                    num_evals=1,
                    initial_op=None,
                    initial_op_feed_dict=None,
                    init_fn=None,
                    eval_op=None,
                    eval_op_feed_dict=None,
                    final_op=None,
                    final_op_feed_dict=None,
                    summary_op=_USE_DEFAULT,
                    summary_op_feed_dict=None,
                    variables_to_restore=None,
                    eval_interval_secs=60,
                    max_number_of_evaluations=None,
                    session_config=None,
                    timeout=None,
                    timeout_fn=None,
                    hooks=None):
  """Runs TF-Slim's Evaluation Loop.
  Args:
    master: The BNS address of the TensorFlow master.
    checkpoint_dir: The directory where checkpoints are stored.
    logdir: The directory where the TensorFlow summaries are written to.
    num_evals: The number of times to run `eval_op`.
    initial_op: An operation run at the beginning of evaluation.
    initial_op_feed_dict: A feed dictionary to use when executing `initial_op`.
    init_fn: An optional callable to be executed after `init_op` is called. The
      callable must accept one argument, the session being initialized.
    eval_op: A operation run `num_evals` times.
    eval_op_feed_dict: The feed dictionary to use when executing the `eval_op`.
    final_op: An operation to execute after all of the `eval_op` executions. The
      value of `final_op` is returned.
    final_op_feed_dict: A feed dictionary to use when executing `final_op`.
    summary_op: The summary_op to evaluate after running TF-Slims metric ops. By
      default the summary_op is set to tf.summary.merge_all().
    summary_op_feed_dict: An optional feed dictionary to use when running the
      `summary_op`.
    variables_to_restore: A list of TensorFlow variables to restore during
      evaluation. If the argument is left as `None` then
      slim.variables.GetVariablesToRestore() is used.
    eval_interval_secs: The minimum number of seconds between evaluations.
    max_number_of_evaluations: the max number of iterations of the evaluation.
      If the value is left as 'None', the evaluation continues indefinitely.
    session_config: An instance of `tf.ConfigProto` that will be used to
      configure the `Session`. If left as `None`, the default will be used.
    timeout: The maximum amount of time to wait between checkpoints. If left as
      `None`, then the process will wait indefinitely.
    timeout_fn: Optional function to call after a timeout.  If the function
      returns True, then it means that no new checkpoints will be generated and
      the iterator will exit.  The function is called with no arguments.
    hooks: A list of additional `SessionRunHook` objects to pass during
      repeated evaluations.
  Returns:
    The value of `final_op` or `None` if `final_op` is `None`.
  """
  if summary_op == _USE_DEFAULT:
    summary_op = summary.merge_all()

  all_hooks = [StopAfterNEvalsHook(num_evals),]

  if summary_op is not None:
    all_hooks.append(SummaryAtEndHook(
        log_dir=logdir, summary_op=summary_op, feed_dict=summary_op_feed_dict))

  if hooks is not None:
    # Add custom hooks if provided.
    all_hooks.extend(hooks)

  saver = None
  if variables_to_restore is not None:
    saver = tf_saver.Saver(variables_to_restore)

  return evaluate_repeatedly(
      checkpoint_dir,
      master=master,
      scaffold=monitored_session.Scaffold(
          init_op=initial_op, init_feed_dict=initial_op_feed_dict,
          init_fn=init_fn, saver=saver),
      eval_ops=eval_op,
      feed_dict=eval_op_feed_dict,
      final_ops=final_op,
      final_ops_feed_dict=final_op_feed_dict,
      eval_interval_secs=eval_interval_secs,
      hooks=all_hooks,
      config=session_config,
      max_number_of_evaluations=max_number_of_evaluations,
      timeout=timeout,
      timeout_fn=timeout_fn)

def checkpoints_iterator(checkpoint_dir,
                         min_interval_secs=0,
                         timeout=None,
                         timeout_fn=None):
  """Continuously yield new checkpoint files as they appear.
  The iterator only checks for new checkpoints when control flow has been
  reverted to it. This means it can miss checkpoints if your code takes longer
  to run between iterations than `min_interval_secs` or the interval at which
  new checkpoints are written.
  The `timeout` argument is the maximum number of seconds to block waiting for
  a new checkpoint.  It is used in combination with the `timeout_fn` as
  follows:
  * If the timeout expires and no `timeout_fn` was specified, the iterator
    stops yielding.
  * If a `timeout_fn` was specified, that function is called and if it returns
    a true boolean value the iterator stops yielding.
  * If the function returns a false boolean value then the iterator resumes the
    wait for new checkpoints.  At this point the timeout logic applies again.
  This behavior gives control to callers on what to do if checkpoints do not
  come fast enough or stop being generated.  For example, if callers have a way
  to detect that the training has stopped and know that no new checkpoints
  will be generated, they can provide a `timeout_fn` that returns `True` when
  the training has stopped.  If they know that the training is still going on
  they return `False` instead.
  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    min_interval_secs: The minimum number of seconds between yielding
      checkpoints.
    timeout: The maximum amount of time to wait between checkpoints. If left as
      `None`, then the process will wait indefinitely.
    timeout_fn: Optional function to call after a timeout.  If the function
      returns True, then it means that no new checkpoints will be generated and
      the iterator will exit.  The function is called with no arguments.
  Yields:
    String paths to latest checkpoint files as they arrive.
  """
  checkpoint_path = None
  while True:
    new_checkpoint_path = wait_for_new_checkpoint(
        checkpoint_dir, checkpoint_path, timeout=timeout)
    if new_checkpoint_path is None:
      if not timeout_fn:
        # timed out
        logging.info('Timed-out waiting for a checkpoint.')
        return
      if timeout_fn():
        # The timeout_fn indicated that we are truly done.
        return
      else:
        # The timeout_fn indicated that more checkpoints may come.
        continue
    start = time.time()
    checkpoint_path = new_checkpoint_path
    yield checkpoint_path
    time_to_next_eval = start + min_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)

def wait_for_new_checkpoint(checkpoint_dir,
                            last_checkpoint=None,
                            seconds_to_sleep=1,
                            timeout=None):
  """Waits until a new checkpoint file is found.
  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    last_checkpoint: The last checkpoint path used or `None` if we're expecting
      a checkpoint for the first time.
    seconds_to_sleep: The number of seconds to sleep for before looking for a
      new checkpoint.
    timeout: The maximum amount of time to wait. If left as `None`, then the
      process will wait indefinitely.
  Returns:
    a new checkpoint path, or None if the timeout was reached.
  """
  logging.info('Waiting for new checkpoint at %s', checkpoint_dir)
  stop_time = time.time() + timeout if timeout is not None else None
  while True:
    checkpoint_path = tf_saver.latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None or checkpoint_path == last_checkpoint:
      if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
        return None
      time.sleep(seconds_to_sleep)
    else:
      logging.info('Found new checkpoint at %s', checkpoint_path)
      return checkpoint_path

def main(unused_argv):

  tf.logging.set_verbosity(tf.logging.INFO)

  """Get dataset-dependent information."""
  # dataset APIで呼んでいる
  dataset = segmentation_dataset.get_dataset(FLAGS.dataset, FLAGS.eval_split, dataset_dir=FLAGS.dataset_dir)
  tf.gfile.MakeDirs(FLAGS.eval_logdir)
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

  with tf.Graph().as_default():
    samples = input_generator.get(
        dataset,
        FLAGS.eval_crop_size,
        FLAGS.eval_batch_size,
        min_resize_value=FLAGS.min_resize_value,
        max_resize_value=FLAGS.max_resize_value,
        resize_factor=FLAGS.resize_factor,
        dataset_split=FLAGS.eval_split,
        is_training=False,
        model_variant=FLAGS.model_variant)

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_classes},
        crop_size=FLAGS.eval_crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    if tuple(FLAGS.eval_scales) == (1.0,):
      tf.logging.info('Performing single-scale test.')
      predictions = model.predict_labels(samples[common.IMAGE], model_options, image_pyramid=FLAGS.image_pyramid)
    else:
      tf.logging.info('Performing multi-scale test.')
      predictions = model.predict_labels_multi_scale(
          samples[common.IMAGE],
          model_options=model_options,
          eval_scales=FLAGS.eval_scales,
          add_flipped_images=FLAGS.add_flipped_images)

    # predictions : The predicted values，Tensor("ArgMax:0", shape=(?, ?, ?), dtype=int64)
    # common.OUTPUT_TYPE = semantic
    predictions = predictions[common.OUTPUT_TYPE]
    # Tensor("Reshape_8:0", shape=(?,), dtype=int64)
    predictions = tf.reshape(predictions, shape=[-1])

    # labels : The ground truth values，Tensor("Reshape_9:0", shape=(?,), dtype=int32)
    labels = tf.reshape(samples[common.LABEL], shape=[-1])

    # weights :
    weights = tf.to_float(tf.not_equal(labels, dataset.ignore_label))

    # ignore_labelかどうかで処理を変える
    # a = tf.equal(labels, dataset.ignore_label) : labelsとdataset.ignore_labelの共通部分にtrueが入る
    # b = tf.zeros_like(labels) : labelsと同じshapeのtensorで各要素は0
    # c = labels
    # tf.where(a, b, c) : aとbとcは同じshape, aのtrueとなるindexの位置を返す
    # aの要素がtrue : b(0)を返す，false : c(そのまま)を返す
    labels = tf.where(tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)

    # """Define the evaluation metric."""
    # predictions_tag = 'miou'
    # for eval_scale in FLAGS.eval_scales:
    #   predictions_tag += '_' + str(eval_scale)
    # if FLAGS.add_flipped_images:
    #   predictions_tag += '_flipped'
    #
    # metric_map = {}
    # # IoU : boxに対して, 目的となる領域(ground truth box)がどれだけ含まれているか
    # metric_map[predictions_tag] = tf.metrics.mean_iou(labels=labels, predictions=predictions, num_classes=dataset.num_classes, weights=weights)
    #
    # # accuracy : 予測がラベルに一致する頻度を計算
    # metric_map["accuracy"] = tf.metrics.accuracy(labels=labels, predictions=predictions, weights=weights)
    #
    # # precision : 適合率
    # # metric_map["precision"] = tf.metrics.precision(labels=labels, predictions=predictions, weights=weights)
    #
    # # mean_per_class_accuracy : クラスごとの精度の平均を計算します
    # metric_map["mean_per_class_accuracy"] = tf.metrics.mean_per_class_accuracy(labels=labels, predictions=predictions, num_classes=dataset.num_classes, weights=weights)
    #
    # metrics_to_values, metrics_to_updates = (tf.contrib.metrics.aggregate_metric_map(metric_map))
    # for metric_name, metric_value in six.iteritems(metrics_to_values):
    #   slim.summaries.add_scalar_summary(metric_value, metric_name, print_summary=True)

    """バッチサイズを設定する"""
    num_batches = int(math.ceil(dataset.num_samples / float(FLAGS.eval_batch_size)))
    tf.logging.info('Eval num images %d', dataset.num_samples) # 1
    tf.logging.info('Eval batch size %d and num batch %d', FLAGS.eval_batch_size, num_batches) # 1492

    # num_eval_itersの設定
    num_eval_iters = None
    if FLAGS.max_number_of_evaluations > 0:
      num_eval_iters = FLAGS.max_number_of_evaluations

    eval_op = [labels, predictions]#.extend(list(metrics_to_updates.values()))

    # モデルを評価する
    evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_logdir,
        num_evals=num_batches,
        eval_op=eval_op,
        max_number_of_evaluations=num_eval_iters,
        eval_interval_secs=FLAGS.eval_interval_secs)

if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('eval_logdir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()


# eval1 (14, 14)
# eval_1 = [[143783, 85, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [7814, 23123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [1876153, 832, 5249, 239, 4137, 1488, 16677, 14683, 5106, 21, 801, 5, 29, 0]]

# (13, 13)
# eval_1 = [[2019936, 917, 5249, 239, 4137, 1488, 16677, 14683, 5106, 21, 801, 5, 29],
#           [7814, 23123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
