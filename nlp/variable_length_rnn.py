# coding: utf-8
"""
https://danijar.com/variable-sequence-lengths-in-tensorflow/
对不定长rnn处理辅助工具
"""
import tensorflow as tf


def length(sequence):
    # 获取序列的真实长度
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def cal_variable_length_rnn():
    # 采用dynamic_rnn进行不定长循环
    max_length = 100
    frame_size = 64
    num_hidden = 200

    sequence = tf.placeholder(
        tf.float32, [None, max_length, frame_size])
    output, state = tf.nn.dynamic_rnn(
        tf.contrib.rnn.GRUCell(num_hidden),
        sequence,
        dtype=tf.float32,
        sequence_length=length(sequence),
    )


def last_relevant(output, length):
    # rnn循环最后一个state
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant
