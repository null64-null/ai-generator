import sys, os
import numpy as np

from propagetion.loss import CrossEntropyError
from learning import Gun
from layer import generate_layers

sys.path.append(os.pardir)
from mnist.mnist import load_mnist


##### input #####
# data
# 最終的に、データセットをまとめたpklファイルを読み込む形にする（以下）
# 入力値は、authのpklファイルパスと、画像サイズとする
'''
# file path, image shape
auth_file_path = 'auth.pkl'           #dummy
auth_image_shape = [5000, 3, 32, 32]  #dummy
'''

# network
# oh = 1 + (h + 2 * pad - fh) / st
error = CrossEntropyError()
lerning_rate = 0.1

generator_layers_params = [
    # (h, w) = (batch_size, 10)
    {
        'layer_type': 'affine_layer',
        'params': {
            'layer_size': [10, 10],
        }
    },
    {
        'layer_type': 'relu',
        'params': {}
    },
    {
        'layer_type': 'verticalize_section',
        'params': {
            'next_layer_size': [100, 1, 5, 5]
        }
    },
    # (h, w) = (batch_size, 1, 5, 5)
    {
        'layer_type': 'deconvolution_layer',
        'params': {
            
        }
    },
    # (h, w) = (batch_size, 1, 28, 28)
]

discriminator_layers_params = [
    # (n, c, h, w) = (batch_size, 1, 28, 28)
    {
        'layer_type': 'convolution_layer',
        'params': {
            'filter_size': [10, 1, 28, 28],
            'pad': 0,
            'st': 1
        }
    },
    # (n, fn, oh, ow) = (batch_size, 10, 1, 1)
    {
        'layer_type': 'relu',
        'params': {}
    },
    {
        'layer_type': 'flatten_section',
        'params': {}
    },
    # (h, w) = (batch_size, 1*1*10) = (batch_size, 10)
    {
        'layer_type': 'affine_layer',
        'params': {
            'layer_size': [10, 1],
        }
    },
    # (h, w) = (batch_size, 1)
    {
        'layer_type': 'soft_max',
        'params': {}
    },
]

# learning, checking setting
iters = 500
picture_check_span = 100


##### process input #####

# dummy mnist
(x_auth, t_auth), (_, _) = load_mnist(one_hot_label=True, normalize=True)
auth_data_length = x_auth.shape[0]
x_auth = x_auth.reshape(auth_data_length, 1, 28, 28)

'''
x_auth, t_auth, _ = generate_data_from_pkl(auth_file_path, auth_image_shape)
'''

data = {
    'x_auth': x_auth,
    't_auth': t_auth,
}

learning_params = {
    'lerning_rate': lerning_rate,
    'iters': iters,
}

checking_params = {
    'picture_check_span': picture_check_span,
}

generator_layers = generate_layers(generator_layers_params)
discriminator_layers = generate_layers(discriminator_layers_params)


##### learn #####
learning = Gun (
    data = data,
    generator_layers = generator_layers,
    discriminator_layers = discriminator_layers,
    error = error,
    learning_params = learning_params,
    checking_params = checking_params,
    is_show_progress = True,
    is_show_pictures = True,
    is_show_result = True,
)

learning.learn()