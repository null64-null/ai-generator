# 1. for mnist (94%)
'''
layers = [
    FlattenSection(),
    AffineLayer([784, 50]),
    Sigmoid(),
    AffineLayer([50, 10]),
    Softmax(),
]
'''

# 2. best for mnist (98%)
'''
# 99%
layers = [
    FlattenSection(), 
    AffineLayer([784, 500]),
    Relu(),
    AffineLayer([500, 100]),
    Relu(),
    AffineLayer([100, 10]),
    Softmax(),
]
'''

# 3. convolution test
'''
layers = [
    # (n, c, h, w) = (batch_size, 1, 28, 28)

    ConvolutionLayer(
        filter_size=[5, 1, 6, 6],
        pad = 0,
        st = 1
    ),
    Relu(),
    # (n, fn, oh, ow) = (batch_size, 5, 23, 23)

    ConvolutionLayer(
        filter_size=[10, 5, 12, 12],
        pad = 0,
        st = 1
    ),
    Relu(),
    # (n, fn, oh, ow) = (batch_size, 10, 12, 12)

    FlattenSection(), 
    # (h, w) = (batch_size, 12*12*10) = (batch_size, 1440)

    AffineLayer([1440, 10]),
    Softmax(),
]
'''

# nomal filter
'''
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
'''

# full size filter
'''
layers = [
    ConvolutionLayer(
        filter_size=[10, 1, 28, 28],
        pad = 0,
        st = 1
    ),
    Relu(), # (n, fn, oh, ow) = (batch_size, 10, 1, 1)
    FlattenSection(), # (h, w) = (batch_size, 1*1*10) = (batch_size, 10)
    AffineLayer([10, 10]),
    Softmax(),
]
'''

# simple affine
'''
layers = [
    FlattenSection(),
    AffineLayer([784, 10]),
    Softmax(),
]
'''
