import numpy as np

def x_init(batch_size, input_size):
    x = np.random.randn(batch_size, input_size)
    return x

def w_init(input_size, output_size):
    w = np.random.randn(input_size, output_size)
    return w

def b_init(output_size):
    b = np.random.randn(output_size)
    return b

def c_init(filter_size):
    c = np.random.randn(filter_size, filter_size)

