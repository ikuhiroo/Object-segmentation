import sys
import tensorflow as tf
from pathlib import Path
from collections import defaultdict

"""
・pascal_vocで使用されている輝度値の確認
"""
# pascolのSegmentationClassの場所
SEGMENTATION = Path("/Users/1-10robotics/Desktop/new")

def read_csv(csvfile):
    fname_queue = tf.train.string_input_producer([csvfile])
    reader = tf.TextLineReader()
    key, val = reader.read(fname_queue)
    fname, label = tf.decode_csv(val, [["aa"], [1]])
    return read_img(fname)

def read_img(fname):
    img_r = tf.read_file(fname)
    return tf.image.decode_image(img_r, channels=1)

def main():
    # 引数設定
    # argv = sys.argv
    # argc = len(argv)
    # if (argc < 2):
    #     print('Usage: python %s csvfile' %argv[0])
    #     quit()
    # image = read_img(argv[1])
    # sess = tf.Session()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # tf.train.start_queue_runners(sess)
    # x = sess.run(image)
    # x_shape = x.shape[0]
    # y_shape = x.shape[1]
    # x = x.flatten().reshape(x_shape, y_shape)
    # print(x.shape)
    # print(x.dtype)
    # print(x)
    #
    # # imgファイルの輝度値を取得する
    # color_value = defaultdict(lambda: 0)
    #
    # for i in range(len(x)):
    #     for j in range(len(x[i])):
    #         color_value[int(x[i][j])] += 1
    # print(color_value)
    """
    ascolのSegmentationClassのファイル名取得
    学習済みモデルに使われている輝度を取得する
    """
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.train.start_queue_runners(sess)
    fname = SEGMENTATION.iterdir()
    for child in fname:
      print(child)
      image = read_img(str(child))
      x = sess.run(image) #np.array
      x_shape = x.shape[0]
      y_shape = x.shape[1]
      x = x.flatten().reshape(x_shape, y_shape)
      for i in range(len(x)):
         for j in range(len(x[i])):
             color_value[int(x[i][j])] += 1

    print(len(color_value))

if __name__ == '__main__':
    main()
