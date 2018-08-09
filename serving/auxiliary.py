# coding: utf-8
import tensorflow as tf
import numpy as np
import tensorflow.contrib.session_bundle.exporter as exporter
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import os


def model_save(sess, path, model_name, global_step):
    """
    模型保存
    :param sess: tf.Session()
    :param path: 保存路径
    :param model_name: 模型名称
    :param global_step: 保存tag
    :return:
    """
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(path, model_name), global_step=global_step)
    return saver


def build_model_decode(saver, sess, work_dir, export_version, inputs, outputs):
    """
    导出模型，tensorflow serving可以直接加载导出的模型
    :param saver: tf.train.Saver()
    :param sess: tf.Session()
    :param work_dir: 导出模型路径
    :param export_version: 版本
    :param inputs: inputs
    :param outputs: outputs
    :return:
    """
    model_exporter = exporter.Exporter(saver)
    model_exporter.init(
        sess.graph.as_graph_def(),
        named_graph_signatures={
            'inputs': exporter.generic_signature({'inputs': inputs}),
            'outputs': exporter.generic_signature({'outputs': outputs})})
    model_exporter.export(work_dir,
                          tf.constant(export_version), sess)


def predict_response_into_nparray(response, output_tensor_name):
    """
    tensorflow serving返回的结果转化成np格式
    """
    dims = response.outputs[output_tensor_name].tensor_shape.dim
    shape = tuple(d.size for d in dims)
    return np.reshape(response.outputs[output_tensor_name].int_val, shape)


def predict_serving(host, port, inputs, model_spec_name):
    """
    seving预测接口
    :param host: IP
    :param port: 端口
    :param inputs: 输入
    :param model_spec_name: 模型名称
    :return:
    """
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_spec_name
    request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(inputs, shape=list(inputs.shape)))
    result = stub.Predict(request, 10.0)
    values = predict_response_into_nparray(result, 'outputs')

