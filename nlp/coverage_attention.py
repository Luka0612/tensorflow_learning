# coding: utf-8
"""
coverage attention
https://arxiv.org/pdf/1601.04811.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from tensorflow import concat
from tensorflow import zeros
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import *
from tensorflow.python.framework import dtypes
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope


def _bahdanau_coverage_score(processed_query, keys, coverage_features, normalize):
    dtype = processed_query.dtype
    # Get the number of hidden units from the trailing dimension of keys
    num_units = keys.shape[2].value or array_ops.shape(keys)[2]
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = array_ops.expand_dims(processed_query, 1)
    coverage_features = array_ops.expand_dims(coverage_features, 1)
    v = variable_scope.get_variable(
        "attention_v", [num_units], dtype=dtype)
    if normalize:
        # Scalar used in weight normalization
        g = variable_scope.get_variable(
            "attention_g", dtype=dtype,
            initializer=math.sqrt((1. / num_units)))
        # Bias added prior to the nonlinearity
        b = variable_scope.get_variable(
            "attention_b", [num_units], dtype=dtype,
            initializer=init_ops.zeros_initializer())
        # normed_v = g * v / ||v||
        normed_v = g * v * math_ops.rsqrt(
            math_ops.reduce_sum(math_ops.square(v)))
        return math_ops.reduce_sum(
            normed_v * math_ops.tanh(keys + processed_query + coverage_features + b), [2])
    else:
        return math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query + coverage_features), [2])


class BahdanauCoverageAttention(BahdanauAttention):
    """
    对BahdanauAttention类增加coverage，其中coverage是采用累加方式
    """

    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=None,
                 dtype=None,
                 name="BahdanauCoverageAttention"):
        super(BahdanauCoverageAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            normalize=normalize,
            probability_fn=probability_fn,
            score_mask_value=score_mask_value,
            dtype=dtype,
            name=name)
        if dtype is None:
            dtype = dtypes.float32
        # coverage状态
        with variable_scope.variable_scope("coverage"):
            self.coverage_layer = layers_core.Dense(
                num_units, name="coverage_layer", use_bias=False, dtype=dtype)
            self.coverage = zeros(shape=[self.batch_size, self._alignments_size], dtype=dtype,
                                  name="coverage")

    def __call__(self, query, state):
        with variable_scope.variable_scope(None, "bahdanau_coverage_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            coverage_features = self.coverage_layer(self.coverage)
            score = _bahdanau_coverage_score(processed_query, self._keys, coverage_features, self._normalize)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        self.coverage += alignments
        return alignments, next_state


class BahdanauRnnCoverageAttention(BahdanauAttention):
    """
    对BahdanauAttention类增加coverage, 其中coverage采用RNN进行更新
    """

    def __init__(self,
                 num_units,
                 memory,
                 coverage_hidden_num_units,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=None,
                 dtype=None,
                 name="BahdanauCoverageAttention"):
        super(BahdanauRnnCoverageAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            normalize=normalize,
            probability_fn=probability_fn,
            score_mask_value=score_mask_value,
            dtype=dtype,
            name=name)
        if dtype is None:
            dtype = dtypes.float32
        # coverage初始状态
        self.coverage_rnn_cell = GRUCell(coverage_hidden_num_units)
        self.coverage_state = self.coverage_rnn_cell.zero_state(self.batch_size, dtype)
        with variable_scope.variable_scope("coverage"):
            self.coverage_layer = layers_core.Dense(
                num_units, name="coverage_layer", use_bias=False, dtype=dtype)

    def __call__(self, query, state):
        with variable_scope.variable_scope(None, "bahdanau_coverage_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            coverage_features = self.coverage_layer(self.coverage_state)
            score = _bahdanau_coverage_score(processed_query, self._keys, coverage_features, self._normalize)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        # 更新coverage_state
        coverage_cell_input = concat([alignments, query], 1)
        _, coverage_cell_state = self.coverage_rnn_cell(coverage_cell_input, self.coverage_state)
        self.coverage_state = coverage_cell_state
        return alignments, next_state


def _bahdanau_coverage_mul_score(processed_query, keys, coverage_features, normalize):
    dtype = processed_query.dtype
    # Get the number of hidden units from the trailing dimension of keys
    num_units = keys.shape[2].value or array_ops.shape(keys)[2]
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = array_ops.expand_dims(processed_query, 1)
    v = variable_scope.get_variable(
        "attention_v", [num_units], dtype=dtype)
    if normalize:
        # Scalar used in weight normalization
        g = variable_scope.get_variable(
            "attention_g", dtype=dtype,
            initializer=math.sqrt((1. / num_units)))
        # Bias added prior to the nonlinearity
        b = variable_scope.get_variable(
            "attention_b", [num_units], dtype=dtype,
            initializer=init_ops.zeros_initializer())
        # normed_v = g * v / ||v||
        normed_v = g * v * math_ops.rsqrt(
            math_ops.reduce_sum(math_ops.square(v)))
        return math_ops.reduce_sum(
            normed_v * math_ops.tanh(keys + processed_query + coverage_features + b), [2])
    else:
        return math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query + coverage_features), [2])


class BahdanauRnnCoverageMulAttention(BahdanauAttention):
    """
    对BahdanauAttention类增加coverage, 其中coverage采用RNN进行更新，每个h，t对应不同的coverage
    https://arxiv.org/pdf/1601.04811.pdf
    """

    def __init__(self,
                 num_units,
                 memory,
                 coverage_hidden_num_units,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=None,
                 dtype=None,
                 name="BahdanauCoverageAttention"):
        super(BahdanauRnnCoverageMulAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            normalize=normalize,
            probability_fn=probability_fn,
            score_mask_value=score_mask_value,
            dtype=dtype,
            name=name)
        if dtype is None:
            dtype = dtypes.float32
        # coverage初始状态
        self.coverage_rnn_cell = GRUCell(coverage_hidden_num_units)
        self.coverage_state = self.coverage_rnn_cell.zero_state(self.batch_size * self._alignments_size, dtype)
        with variable_scope.variable_scope("coverage"):
            self.coverage_layer = layers_core.Dense(
                num_units, name="coverage_layer", use_bias=False, dtype=dtype)

    def __call__(self, query, state):
        with variable_scope.variable_scope(None, "bahdanau_coverage_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            coverage_features = self.coverage_layer(self.coverage_state)
            coverage_features = array_ops.reshape(coverage_features, [self.batch_size, self._alignments_size, -1])
            score = _bahdanau_coverage_mul_score(processed_query, self._keys, coverage_features, self._normalize)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        # 更新coverage_state
        coverage_cell_input = concat([alignments, query], 1)
        # coverage_cell_input复制alignments_size份
        coverage_cell_input_tile = tf.contrib.seq2seq.tile_batch(coverage_cell_input, multiplier=self._alignments_size)
        # 将value reshape
        coverage_value_reshape = array_ops.reshape(self.values, [self.batch_size * self._alignments_size, -1])
        coverage_cell_input_tile = concat([coverage_cell_input_tile, coverage_value_reshape], 1)
        _, coverage_cell_state = self.coverage_rnn_cell(coverage_cell_input_tile, self.coverage_state)
        self.coverage_state = coverage_cell_state
        return alignments, next_state


def _luong_coverage_score(query, keys, coverage_features, scale):
    """Implements Luong-style (multiplicative) scoring function.

    This attention has two forms.  The first is standard Luong attention,
    as described in:

    Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
    "Effective Approaches to Attention-based Neural Machine Translation."
    EMNLP 2015.  https://arxiv.org/abs/1508.04025

    The second is the scaled form inspired partly by the normalized form of
    Bahdanau attention.

    To enable the second form, call this function with `scale=True`.

    Args:
      query: Tensor, shape `[batch_size, num_units]` to compare to keys.
      keys: Processed memory, shape `[batch_size, max_time, num_units]`.
      scale: Whether to apply a scale to the score function.

    Returns:
      A `[batch_size, max_time]` tensor of unnormalized score values.

    Raises:
      ValueError: If `key` and `query` depths do not match.
    """
    depth = query.get_shape()[-1]
    key_units = keys.get_shape()[-1]
    if depth != key_units:
        raise ValueError(
            "Incompatible or unknown inner dimensions between query and keys.  "
            "Query (%s) has units: %s.  Keys (%s) have units: %s.  "
            "Perhaps you need to set num_units to the keys' dimension (%s)?"
            % (query, depth, keys, key_units, key_units))
    dtype = query.dtype

    # Reshape from [batch_size, depth] to [batch_size, 1, depth]
    # for matmul.
    query = array_ops.expand_dims(query, 1)

    keys = math_ops.matmul(keys, coverage_features, transpose_a=True)
    score = math_ops.matmul(query, keys)
    score = array_ops.squeeze(score, [1])

    if scale:
        # Scalar used in weight scaling
        g = variable_scope.get_variable(
            "attention_g", dtype=dtype,
            initializer=init_ops.ones_initializer, shape=())
        score = g * score
    return score


class LuongCoverageAttention(LuongAttention):
    """
    对LuongAttention增加coverage attention，可以认为coverage对score增加权重使得之前coverage较高的h，score相应减少
    """

    def __init__(self,
                 num_units,
                 memory,
                 coverage_hidden_num_units,
                 memory_sequence_length=None,
                 scale=False,
                 probability_fn=None,
                 score_mask_value=None,
                 dtype=None,
                 name="LuongAttention"):
        super(LuongCoverageAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            scale=scale,
            probability_fn=probability_fn,
            score_mask_value=score_mask_value,
            dtype=dtype,
            name=name
        )
        if dtype is None:
            dtype = dtypes.float32
        # coverage初始状态
        self.coverage_rnn_cell = GRUCell(coverage_hidden_num_units)
        self.coverage_state = self.coverage_rnn_cell.zero_state(self.batch_size * self._alignments_size, dtype)
        with variable_scope.variable_scope("coverage"):
            self.coverage_layer = layers_core.Dense(
                self._alignments_size, name="coverage_layer", use_bias=False, dtype=dtype)

    def __call__(self, query, state):
        with variable_scope.variable_scope(None, "luong_attention", [query]):
            coverage_features = self.coverage_layer(self.coverage_state)
            coverage_features = array_ops.reshape(coverage_features, [self.batch_size, self._alignments_size, -1])
            score = _luong_coverage_score(query, self._keys, coverage_features, self._scale)
        alignments = self._probability_fn(score, state)
        next_state = alignments
        # 更新coverage_state
        coverage_cell_input = concat([alignments, query], 1)
        # coverage_cell_input复制alignments_size份
        coverage_cell_input_tile = tf.contrib.seq2seq.tile_batch(coverage_cell_input, multiplier=self._alignments_size)
        # 将value reshape
        coverage_value_reshape = array_ops.reshape(self.values, [self.batch_size * self._alignments_size, -1])
        coverage_cell_input_tile = concat([coverage_cell_input_tile, coverage_value_reshape], 1)
        _, coverage_cell_state = self.coverage_rnn_cell(coverage_cell_input_tile, self.coverage_state)
        self.coverage_state = coverage_cell_state
        return alignments, next_state
