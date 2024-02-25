import os
import numpy as np
import time
from network.network import FlattenSection
from network.pooling import MaxPooling, AveragePooling

x = np.array([
    [
        [
            [11,12,13,21,22,23],
            [14,15,16,24,25,26],
            [17,18,19,27,28,29],
            [31,32,33,400,42,43],
            [34,35,36,44,45,46],
            [37,38,39,47,48,49]
        ],
    ],
])

dx_next = np.array([
    [
        [
            [51,52],
            [53,54]
        ],
    ],
])


class NextLayer:
    def __init__(self, x):
        self.x = x
        self.dx = dx_next

layer = AveragePooling(3)
x_next = layer.func(x)

layer_prev = NextLayer(x_next)
layer.generate_grad(layer_prev)

print("=======")
print(layer_prev.x)
print(layer.dx)

a = [
    {
        'a': 'aaa',
        'b': [1,2,3]
    },
    {
        'a': 'aaa2',
        'b': [1,2,4]
    },
]

print(a[1]['b'])
