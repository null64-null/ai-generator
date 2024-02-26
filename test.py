import os
import numpy as np
import time
from network.network import FlattenSection, DeconvolutionLayer
from network.pooling import MaxPooling, AveragePooling, NNUnpooling

'''
dx_next = np.array([
    [
        [
            [11,11,12,12],
            [11,11,12,12],
            [13,13,14,14],
            [13,13,14,14],
        ],
        [
            [21,21,22,22],
            [21,21,22,22],
            [23,23,24,24],
            [23,23,24,24],
        ],
    ],
])

x = np.array([
    [
        [
            [11,12],
            [13,14]
        ],
        [
            [21,22],
            [23,24]
        ],
    ],
])


class NextLayer:
    def __init__(self, x):
        self.x = x
        self.dx = dx_next

layer = NNUnpooling(2)
x_next = layer.func(x)

layer_prev = NextLayer(x_next)
layer.generate_grad(layer_prev)

print("=======")
print(layer_prev.x)
print(layer.dx)
'''

dx_next = np.array([
    [
        [
            [11,12,13,0],
            [14,15,16,0],
            [17,18,19,0],
            [0,0,0,0],
        ],
    ],
])

x = np.array([
    [
        [
            [11,12,13],
            [14,15,16],
            [17,18,19],
        ],
        [
            [21,22,23],
            [24,25,26],
            [27,28,29],
        ],
    ],
])

w = np.array([
    [
        [
            [11,12],
            [13,14]
        ],
        [
            [21,22],
            [23,24]
        ],
    ],
])

b = np.array([1,1])

print("=======")

class NextLayer:
    def __init__(self, x):
        self.x = x
        self.dx = dx_next

layer = DeconvolutionLayer([1, 2, 2, 2], 0, 1)
x_next = layer.func(x)
print(x_next)

print("=======")

layer_prev = NextLayer(x_next)
layer.generate_grad(layer_prev)
print(layer.dx)





'''
x = np.array([
    [11,12,13,14,21,22,23,24],
    [15,16,17,18,25,26,27,28],
    [31,32,33,34,41,42,43,44],
    [35,36,37,38,45,46,47,48],
])

n = 2
fn = 2
fhfw = 2
hw = 4

x_com = x.reshape(fn, n, fhfw, hw).transpose(0,2,1,3)
print(x_com)


a = np.array([
    [11,12],
    [13,14],
    [21,22],
    [24,25],
])

n = 1
fn = 2
fhfw = 2
hw = 2

print(a.reshape(fn, n, fhfw, hw).shape)
a_com = a.reshape(fn, n, fhfw, hw).transpose(0,2,1,3).reshape(n, fn, fhfw, hw)
print(a_com)
'''