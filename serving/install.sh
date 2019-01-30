
# https://www.tensorflow.org/serving/setup
# 分别安装bazel(compiling source code)、gRPC、TensorFlow Serving dependencies、TensorFlow Serving Python API PIP package

# 源码安装
git clone --recurse-submodules https://github.com/tensorflow/serving

# https://my.oschina.net/jxcdwangtao/blog/1585363 加上对应的copt选项进行编译
bazel build -c opt --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-O3 tensorflow_serving/...

# https://my.oschina.net/jxcdwangtao/blog/1585363 相关参数解释
# https://github.com/tensorflow/serving/issues/344 参数调优
# https://bbs.aliyun.com/read/563646.html?spm=5176.10695662.1996646101.searchclickresult.49074655qy58u0 阿里云集群部署
# https://blog.csdn.net/y19930105/article/details/80763467 采用docker安装，推荐！