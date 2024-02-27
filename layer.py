from network.network import AffineLayer, ConvolutionLayer, FlattenSection, VerticalizeSection, DeconvolutionLayer
from network.pooling import MaxPooling, AveragePooling, NNUnpooling
from network.activation import Relu, LeakyRelu, Sigmoid, Tanh, Softmax

def layer(layer_type, params):
    if layer_type == 'affine_layer':
        layer = AffineLayer(layer_sizes = params['layer_sizes'])
        return layer
    if layer_type == 'convolution_layer':
        layer = ConvolutionLayer(filter_size=params['filter_size'], pad=params['pad'], st=params['st'])
        return layer
    if layer_type == 'deconvolution_layer':
        layer = DeconvolutionLayer(filter_size=params['filter_size'], pad=params['pad'], st=params['st'])
        return layer
    if layer_type == 'flatten_section':
        layer = FlattenSection()
        return layer
    if layer_type == 'verticalize_section':
        layer = VerticalizeSection(next_layer_size = params['next_layer_size'])
        return layer
    if layer_type == 'max_pooling':
        layer = MaxPooling(st=params['st'])
        return layer
    if layer_type == 'average_pooling':
        layer = AveragePooling(st=params['st'])
        return layer
    if layer_type == 'nn_unpooling':
        layer = NNUnpooling(st=params['st'])
        return layer
    if layer_type == 'relu':
        layer = Relu()
        return layer
    if layer_type == 'leaky_relu':
        layer = LeakyRelu(grad=params['grad'])
        return layer
    if layer_type == 'sigmoid':
        layer = Sigmoid()
        return layer
    if layer_type == 'tanh':
        layer = Tanh()
        return layer
    if layer_type == 'soft_max':
        layer = Softmax()
        return layer
    
def generate_layers(layer_params):
    layers = []
    for layer_param in layer_params:
        layer_type = layer_param['layer_type']
        params = layer_param['params']
        layers.append(layer(layer_type, params))
    return layers

'''
layer_params = [
    {
        'layer_type': 'convolution_layer',
        'params': {
            'filter_size': [5, 1, 6, 6],
            'pad': 0,
            'st': 1
        }
    },
    {
        'layer_type': 'relu',
        'params': None
    },
    {
        'layer_type': 'convolution_layer',
        'params': {
            'filter_size': [10, 5, 12, 12],
            'pad': 0,
            'st': 1
        }
    },
    {
        'layer_type': 'relu',
        'params': None
    },
    {
        'layer_type': 'flatten_section',
        'params': None
    },
    {
        'layer_type': 'affine_layer',
        'params': {
            'layer_size': [1440, 10],
        }
    },
    {
        'layer_type': 'softmax',
        'params': None
    },
]
'''
