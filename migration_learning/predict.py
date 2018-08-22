# -*- coding: utf-8 -*-
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Inception-v3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

# Inception-v3模型中代表瓶颈层结果的张量名称。
# 在谷歌提出的Inception-v3模型中，这个张量名称就是'pool_3/_reshape:0'。
# 在训练模型时，可以通过tensor.name来获取张量的名称。
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# 图像输入张量所对应的名称。
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# 下载的谷歌训练好的Inception-v3模型文件目录
MODEL_DIR = 'model/'

# 下载的谷歌训练好的Inception-v3模型文件名
MODEL_FILE = 'tensorflow_inception_graph.pb'

# 因为一个训练数据会被使用多次，所以可以将原始图像通过Inception-v3模型计算得到的特征向量保存在文件中，免去重复的计算。
# 下面的变量定义了这些文件的存放地址。
# CACHE_DIR = 'tmp/bottleneck/'
CACHE_DIR = 'tmp/xiaohongshu/'

# 图片数据文件夹。
# 在这个文件夹中每一个子文件夹代表一个需要区分的类别，每个子文件夹中存放了对应类别的图片。
# INPUT_DATA = 'flower_data/'
INPUT_DATA = 'xiaohongshu/'

# 验证的数据百分比
VALIDATION_PERCENTAGE = 10
# 测试的数据百分比
TEST_PERCENTAGE = 10

# 定义神经网络的设置
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

# 模型保存位置
save_path = 'model_save/xiaohongshu/v1/'
save_name = 'xiaohongshu'


# 这个函数使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量。
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    # 这个过程实际上就是将当前图片作为输入计算瓶颈张量的值。这个瓶颈张量的值就是这张图片的新的特征向量。
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 经过卷积神经网络处理的结果是一个四维数组，需要将这个结果压缩成一个特征向量（一维数组）
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def import_tensor_from_inception_v3(graph_def):
    # 读取已经训练好的Inception-v3模型。
    # 谷歌训练好的模型保存在了GraphDef Protocol Buffer中，里面保存了每一个节点取值的计算方法以及变量的取值。
    # TensorFlow模型持久化的问题在第5章中有详细的介绍。
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def.ParseFromString(f.read())
    # 加载读取的Inception-v3模型，并返回数据输入所对应的张量以及计算瓶颈层结果所对应的张量。
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME,
                                                                                          JPEG_DATA_TENSOR_NAME])
    return bottleneck_tensor, jpeg_data_tensor


class Model():
    def __init__(self):
        n_classes = 3
        # 定义一层全连接层来解决新的图片分类问题。
        # 因为训练好的Inception-v3模型已经将原始的图片抽象为了更加容易分类的特征向量了，所以不需要再训练那么复杂的神经网络来完成这个新的分类任务。
        with tf.name_scope('final_training_ops'):
            weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
            biases = tf.Variable(tf.zeros([n_classes]))
            self.weights = weights
            self.biases = biases

    def predict(self, bottlenecks):
        bottlenecks = tf.constant(bottlenecks, dtype=tf.float32)
        logits = tf.matmul(bottlenecks, self.weights) + self.biases
        final_tensor = tf.nn.softmax(logits)
        return tf.argmax(final_tensor, 1)


def restore(path):
    graph = tf.Graph()
    with graph.as_default():
        m = Model()
        saver = tf.train.Saver()
        session = tf.Session()
        model_file = tf.train.latest_checkpoint(path)
        saver.restore(session, model_file)
        return m, graph, session


graph_inception = tf.Graph()
session_inception = tf.Session()
bottleneck_tensor, jpeg_data_tensor = import_tensor_from_inception_v3(graph_inception.as_graph_def())

model, graph_pre, session_pre = restore(save_path)


def pre_image_class(image_path):

    image_data = gfile.FastGFile(image_path, 'rb').read()
    # 通过Inception-v3模型计算特征向量
    with graph_inception.as_default():
        bottleneck_values = run_bottleneck_on_image(session_inception, image_data, jpeg_data_tensor, bottleneck_tensor)
    bottlenecks = [bottleneck_values]
    with graph_pre.as_default():
        pre_label = model.predict(np.array(bottlenecks))
        return session_pre.run(pre_label)


def pre_images(image_paths):
    bottlenecks = []
    # inception_v3特征
    for image_path in image_paths:
        image_data = gfile.FastGFile(image_path, 'rb').read()
        # 通过Inception-v3模型计算特征向量
        with graph_inception.as_default():
            bottleneck_values = run_bottleneck_on_image(session_inception, image_data, jpeg_data_tensor,
                                                        bottleneck_tensor)
            bottlenecks.append(bottleneck_values)
    # 全连接层
    with graph_pre.as_default():
        pre_label = model.predict(np.array(bottlenecks))
        return session_pre.run(pre_label)


if __name__ == '__main__':
    print pre_image_class("xiaohongshu/photo/1_1.jpg")
    print pre_images(["xiaohongshu/photo/1_1.jpg",
                      "xiaohongshu/photo/1fe05b75-8156-484b-aafc-c252028d6cc9@r_750w_750h_ss1.jpg",
                      "xiaohongshu/excel/0b5a478d-8862-4818-a2f7-d6173bf81011@r_750w_750h_ss1.jpg",
                      "xiaohongshu/writing/4b05153c-a508-4875-977c-4453aad7864d@r_750w_750h_ss1.jpg"])