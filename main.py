import sys, os
import numpy as np

from network.activation import Relu, Sigmoid, Tanh, Softmax
from network.network import AffineLayer, ConvolutionLayer, FlattenSection
from network.pooling import MaxPooling, AveragePooling
from propagetion.loss import CrossEntropyError
from learning import Supervised_learning
from util.data_processing import generate_data_from_pkl

sys.path.append(os.pardir)
from mnist.mnist import load_mnist

# data
# 最終的に、ここはデータセットをまとめたpklファイルを読み込む形にする（以下）
# 入力値は、trainとtestのpklファイルパスと、画像サイズとする
'''
train_file_path = 'train.pkl'           #dummy
train_image_shape = [5000, 3, 32, 32]   #dummy
test_file_path = 'test.pkl'             #dummy
test_image_shape = [1000, 3, 32, 32]    #dummy

x_train, t_train, _ = generate_data_from_pkl(train_file_path, train_image_shape)
x_test, t_test, _ = generate_data_from_pkl(test_file_path, test_image_shape)
'''

(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True, normalize=True)
train_data_length = x_train.shape[0]
test_data_length = x_test.shape[0]
x_train = x_train.reshape(train_data_length, 1, 28, 28)
x_test = x_test.reshape(test_data_length, 1, 28, 28)

# batch size
batch_size = 100

### network ###
error = CrossEntropyError()
lerning_rate = 0.1

# layers
# st = 1, pad = 0
# oh = 1 + (h + 2 * pad - fh) / st
# ow = 1 + (w + 2 * pad - fw) / st
# (n, c, h, w) = (batch_size, 1, 28, 28)

layers = [
    ConvolutionLayer(
        filter_size=[10, 1, 28, 28],
        pad = 0,
        st = 1
    ),
    Relu(), # (n, fn, oh, ow) = (batch_size, 10, 1, 1)
    FlattenSection(), # (h, w) = (batch_size, 1*1*10) = (batch_size, 10)
    #AffineLayer([10, 10]),
    Softmax(),
]

# learning, checking setting
iters = 10000
accuracy_check_span = 1000
check_mask_size_train = 50
check_mask_size_test = 50

data = {
    'x_train': x_train,
    't_train': t_train,
    'x_test': x_test,
    't_test': t_test,
}

learning_params = {
    'lerning_rate': lerning_rate,
    'batch_size': batch_size,
    'iters': iters,
}

checking_params = {
    'accuracy_check_span': accuracy_check_span,
    'check_mask_size': 50,
}

learning = Supervised_learning (
    data = data,
    layers = layers,
    error = error,
    learning_params = learning_params,
    checking_params = checking_params,
    isShowProgress = True,
    isShowGraph = True,
    isShowResult = True,
    isShowFilters = True,
)

learning.learn()