import sys, os
import numpy as np

from network.activation import Relu, Sigmoid, Tanh, Softmax
from network.network import AffineLayer, ConvolutionLayer, FlattenSection
from network.pooling import MaxPooling, AveragePooling
from propagetion.loss import CrossEntropyError
from learning import Supervised_learning

sys.path.append(os.pardir)
from mnist.mnist import load_mnist

# data
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
        filter_size=[5, 1, 8, 8],
        pad = 0,
        st = 1
    ),
    Relu(), # (n, fn, oh, ow) = (batch_size, 5, 21, 21)
    MaxPooling(st=3),
    Relu(), # (n, fn, oh, ow) = (batch_size, 5, 7, 7)
    FlattenSection(), # (h, w) = (batch_size, 7*7*5) = (batch_size, 245)
    AffineLayer([245, 10]),
    Softmax(),
]

# learning, checking setting
iters = 1000
accuracy_check_span = 50
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