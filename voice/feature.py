# coding: utf-8
# 语音处理与特征提取
import tensorflow as tf

# 信号窗口处理与重叠
# 在有些语音识别中，可以采用一些trick，如：stack-4_downdownsampled-3
# A batch of float32 time-domain signals in the range [-1, 1] with shape
# [batch_size, signal_length]. Both batch_size and signal_length may be unknown.
signals = tf.placeholder(tf.float32, [None, None])

# Compute a [batch_size, ?, 128] tensor of fixed length, overlapping windows
# where each window overlaps the previous by 75% (frame_length - frame_step
# samples of overlap).
frames = tf.contrib.signal.frame(signals, frame_length=128, frame_step=32)

# `magnitude_spectrograms` is a [batch_size, ?, 129] tensor of spectrograms. We
# would like to produce overlapping fixed-size spectrogram patches; for example,
# for use in a situation where a fixed size input is needed.
magnitude_spectrograms = tf.abs(tf.contrib.signal.stft(
    signals, frame_length=256, frame_step=64, fft_length=256))

# `spectrogram_patches` is a [batch_size, ?, 64, 129] tensor containing a
# variable number of [64, 129] spectrogram patches per batch item.
spectrogram_patches = tf.contrib.signal.frame(
    magnitude_spectrograms, frame_length=64, frame_step=16, axis=1)


# 重建信号：STFT
# Reconstructs `signals` from `frames` produced in the above example. However,
# the magnitude of `reconstructed_signals` will be greater than `signals`.
reconstructed_signals = tf.contrib.signal.overlap_and_add(frames, frame_step=32)
frame_length = 128
frame_step = 32
windowed_frames = frames * tf.contrib.signal.hann_window(frame_length)
reconstructed_signals = tf.contrib.signal.overlap_and_add(
    windowed_frames, frame_step)


# 计算频谱图
# A batch of float32 time-domain signals in the range [-1, 1] with shape
# [batch_size, signal_length]. Both batch_size and signal_length may be unknown.
signals = tf.placeholder(tf.float32, [None, None])

# `stfts` is a complex64 Tensor representing the Short-time Fourier Transform of
# each signal in `signals`. Its shape is [batch_size, ?, fft_unique_bins]
# where fft_unique_bins = fft_length // 2 + 1 = 513.
stfts = tf.contrib.signal.stft(signals, frame_length=1024, frame_step=512,
                               fft_length=1024)

# A power spectrogram is the squared magnitude of the complex-valued STFT.
# A float32 Tensor of shape [batch_size, ?, 513].
power_spectrograms = tf.real(stfts * tf.conj(stfts))

# An energy spectrogram is the magnitude of the complex-valued STFT.
# A float32 Tensor of shape [batch_size, ?, 513].
magnitude_spectrograms = tf.abs(stfts)
log_offset = 1e-6
log_magnitude_spectrograms = tf.log(magnitude_spectrograms + log_offset)


# 计算log_mel
# Warp the linear-scale, magnitude spectrograms into the mel-scale.
num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
sample_rate = 16000
lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 64
linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
  num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
  upper_edge_hertz)
mel_spectrograms = tf.tensordot(
  magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
# Note: Shape inference for <a href="../../api_docs/python/tf/tensordot"><code>tf.tensordot</code></a> does not currently handle this case.
mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
  linear_to_mel_weight_matrix.shape[-1:]))
log_offset = 1e-6
log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)

# 计算mfcc
num_mfccs = 13
# Keep the first `num_mfccs` MFCCs.
mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
    log_mel_spectrograms)[..., :num_mfccs]
