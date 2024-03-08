import sys, os
import numpy as np

from propagetion.loss import CrossEntropyError
from learning import Supervised_learning
from layer import generate_layers

sys.path.append(os.pardir)
from mnist.mnist import load_mnist


##### input #####
# data
# 最終的に、データセットをまとめたpklファイルを読み込む形にする（以下）
# 入力値は、trainとtestのpklファイルパスと、画像サイズとする
'''
# file path, image shape
train_file_path = 'train.pkl'           #dummy
train_image_shape = [5000, 3, 32, 32]   #dummy
test_file_path = 'test.pkl'             #dummy
test_image_shape = [1000, 3, 32, 32]    #dummy
'''

# batch size
batch_size = 100

# network
# oh = 1 + (h + 2 * pad - fh) / st
error = CrossEntropyError()
lerning_rate = 0.1
layer_params = [
    # (n, c, h, w) = (batch_size, 1, 28, 28)
    {
        'layer_type': 'convolution_layer',
        'params': {
            'filter_size': [5, 1, 7, 7],
            'pad': 0,
            'st': 1
        }
    },
    # (n, fn, oh, ow) = (batch_size, 5, 22, 22)
    {
        'layer_type': 'relu',
        'params': {}
    },
    {
        'layer_type': 'flatten_section',
        'params': {}
    },
    # (h, w) = (batch_size, 1*1*10) = (batch_size, 2420)
    {
        'layer_type': 'affine_layer',
        'params': {
            'layer_sizes': [2420, 10]
        }
    },
    {
        'layer_type': 'soft_max',
        'params': {}
    },
]

# learning, checking setting
iters = 1000
accuracy_check_span = 200
check_mask_size_train = 50
check_mask_size_test = 50


##### process input #####

# dummy mnist
(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True, normalize=True)
train_data_length = x_train.shape[0]
test_data_length = x_test.shape[0]
x_train = x_train.reshape(train_data_length, 1, 28, 28)
x_test = x_test.reshape(test_data_length, 1, 28, 28)
'''
x_train, t_train, _ = generate_data_from_pkl(train_file_path, train_image_shape)
x_test, t_test, _ = generate_data_from_pkl(test_file_path, test_image_shape)
'''

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

layers = generate_layers(layer_params)


##### learn #####
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