# coding: utf-8
import tensorflow as tf

"""
获取单个operation/variable
可以通过如下两个方法获取图中的相关variable和operation：
1. tf.Graph.get_tensor_by_name(tensor_name)
2. tf.Graph.get_operation_by_name(op_name)
"""

# 批量获取
# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)
v3 = tf.get_variable("v3", shape=[4], initializer=tf.zeros_initializer)

inc_v1 = tf.assign(v1, v1 + 1, name='inc_v1')
dec_v2 = tf.assign(v2, v2 - 1, name='dec_v2')
dec_v3 = tf.assign(v3, v3 - 2, name='dec_v3')

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
    sess.run(init_op)
    # Do some work with the model.
    inc_v1.op.run()
    dec_v2.op.run()
    dec_v3.op.run()

    # 列出了每个graph中每个node的详细信息
    for n in tf.get_default_graph().as_graph_def().node:
        print n

    # op.valuses()将返回该op对应的tensor对象，可以进一步获取tensor的name，shape等信息。
    for op in tf.get_default_graph().get_operations():
        print op.name
        print op.values()
        """
        name:v1/Initializer/zeros
        value:(<tf.Tensor 'v1/Initializer/zeros:0' shape=(3,) dtype=float32>,)
        name:v1
        value:(<tf.Tensor 'v1:0' shape=(3,) dtype=float32_ref>,)
        """

    # 该方法返回默认计算图中所有的variable()对象
    for variable in tf.all_variables():
        print variable
        print variable.name
        """
        <tf.Variable 'v1:0' shape=(3,) dtype=float32_ref>
        v1:0
        <tf.Variable 'v2:0' shape=(5,) dtype=float32_ref>
        v2:0
        <tf.Variable 'v3:0' shape=(4,) dtype=float32_ref>
        v3:0
        """

    # 根据key返回相应collection中的对象
    for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        print variable
        """
        <tf.Variable 'v1:0' shape=(3,) dtype=float32_ref>
        <tf.Variable 'v2:0' shape=(5,) dtype=float32_ref>
        <tf.Variable 'v3:0' shape=(4,) dtype=float32_ref>
        """

    # Add variable into
    tf.add_to_collection('test', v1)
    tf.add_to_collection('test', v2)
    tf.add_to_collection('test', inc_v1)

    for element in tf.get_collection('test'):
        print element
        """
        <tf.Variable 'v1:0' shape=(3,) dtype=float32_ref>
        <tf.Variable 'v2:0' shape=(5,) dtype=float32_ref>
        Tensor("inc_v1:0", shape=(3,), dtype=float32_ref)
        """

    # 获取graph中所有collection
    for key in tf.get_default_graph().get_all_collection_keys():
        print 'key:' + key
        for element in tf.get_collection(key):
            print element

        """
        key:variables
        <tf.Variable 'v1:0' shape=(3,) dtype=float32_ref>
        <tf.Variable 'v2:0' shape=(5,) dtype=float32_ref>
        <tf.Variable 'v3:0' shape=(4,) dtype=float32_ref>
        key:trainable_variables
        <tf.Variable 'v1:0' shape=(3,) dtype=float32_ref>
        <tf.Variable 'v2:0' shape=(5,) dtype=float32_ref>
        <tf.Variable 'v3:0' shape=(4,) dtype=float32_ref>
        """