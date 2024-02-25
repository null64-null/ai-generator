import os
import numpy as np
import time
from network.network import FlattenSection
from network.pooling import MaxPooling, AveragePooling

'''
x = np.array([
    [
        [
            [11,12,13,0],
            [14,15,16,0],
            [17,18,19,0],
            [0,0,0,0],
        ],
        [
            [21,22,23,0],
            [24,25,26,0],
            [27,28,29,0],
            [0,0,0,0],
        ],
          [
            [31,32,33,0],
            [34,35,36,0],
            [37,38,39,100],
            [0,0,0,0],
        ],
    ],
    [
        [
            [41,42,43,0],
            [44,45,46,0],
            [47,48,49,0],
            [0,0,0,0],
        ],
        [
            [51,52,53,0],
            [54,55,56,0],
            [57,58,59,0],
            [0,0,0,0],
        ],
          [
            [61,62,63,0],
            [64,65,66,0],
            [67,68,69,0],
            [0,0,0,0],
        ],
    ],
])


dx_next = np.array([
    [
        [
            [11,12],
            [13,14],
        ],
        [
            [21,22],
            [23,24],
        ],
          [
            [31,32],
            [33,34],
        ],
    ],
    [
        [
            [41,42],
            [43,44],
        ],
        [
            [51,52],
            [53,54],
        ],
          [
            [61,62],
            [63,64],
        ],
    ],
])
'''
'''
x = np.array([
    [
        [
            [11,12],
            [13,14],
        ],
        [
            [21,22],
            [23,24],
        ],
          [
            [31,32],
            [33,34],
        ],
    ],
    [
        [
            [41,42],
            [43,44],
        ],
        [
            [51,52],
            [53,54],
        ],
          [
            [61,62],
            [63,64],
        ],
    ],
])

dx_next = np.array([
    [11,12,13,14,21,22,23,24,31,32,33,34],
    [41,42,43,44,51,52,53,54,61,62,63,64],
])
'''

'''
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
'''


'''
dw = np.array([
    [
        [
            [11,12],
            [13,14]
        ],
        [
            [21,22],
            [23,24]
        ],
        [
            [31,32],
            [33,34]
        ],
    ],
    [
        [
            [41,42],
            [43,44]
        ],
        [
            [51,52],
            [53,54]
        ],
        [
            [61,62],
            [63,64]
        ],
    ],
])

dw_com = np.array([
    [11,21,31],
    [12,22,32],
    [13,23,33],
    [14,24,34],
    [41,51,61],
    [42,52,62],
    [43,53,63],
    [44,54,64],
])

dx = dw
dx_com = dw_com.T

n = 2
c = 3
h = 2
w = 2
fn = 2
fh = 2
fw = 2

dw = dw_com.reshape(fn, fh*fw, c).transpose(0, 2, 1).reshape(fn, c, fh, fw)
print(dw)

dx = dx_com.T.reshape(n, h*w, c).transpose(0, 2, 1).reshape(n, c, h, w)
print(dx)

#n, c, h, w = x.shape
#x_flatten = x.reshape(n, c, 1, h*w).transpose(1, 0, 2, 3).reshape(c, n*h*w)
#print(x_flatten)

#fn, c, fh, fw = w.shape
#w_com = w.reshape(fn, c, fh*fw).transpose(0, 2, 1).reshape(fn*fh*fw, c)
#print(w_com)

'''




'''
z = np.array([
    [11,12,13,41,42,43],
    [14,15,16,44,45,46],
    [21,22,23,51,52,53],
    [24,25,26,54,55,56],
    [31,32,33,61,62,63],
    [34,35,36,64,65,66],
])

dz = np.array([
    [
        [
            [11,12,13],
            [14,15,16]
        ],
        [
            [21,22,23],
            [24,25,26]
        ],
        [
            [31,32,33],
            [34,35,36]
        ],
    ],
    [
        [
            [41,42,43],
            [44,45,46]
        ],
        [
            [51,52,53],
            [54,55,56]
        ],
        [
            [61,62,63],
            [64,65,66]
        ],
    ],
])

n = 2
c = 3
oh = 2
ow = 3

z = z.reshape(c, n, oh, ow).transpose(0,2,1,3).transpose(1, 0, 2, 3)
print(z)
dz = dz.transpose(1, 0, 2, 3).transpose(0,2,1,3).reshape(6, 6)
print(dz)
'''

x = np.array([
    [
        [
            [11,12],
            [13,14],
        ],
        [
            [21,22],
            [23,24],
        ],
          [
            [31,32],
            [33,34],
        ],
    ],
])

dx = np.array([
    [
        [
            [11, 11, 12, 12],
            [11, 11, 12, 12],
            [13, 13, 14, 14],
            [13, 13, 14, 14],
        ],
        [
            [21, 21, 22, 22],
            [21, 21, 22, 22],
            [23, 23, 24, 24],
            [23, 23, 24, 24],
        ],
          [
            [31, 31, 32, 32],
            [31, 31, 32, 32],
            [33, 33, 34, 34],
            [33, 33, 34, 34],
        ],
    ],
])

n = 1
c = 3
h = 2
w = 2

ow = 4
oh = 4

st = 2

oh = h * st
ow = w * st

'''
a = np.ones((n, c, oh, ow))
a = a.reshape(n, c, h, w, st, st)

x = x.reshape(n, c, h, w, 1, 1)

x_next = a*x

x_next = x_next.transpose(0,1,2,4,3,5).reshape(n, c, oh, ow)

print(x_next)
'''

dx = dx.reshape(n, c, h, w, st, st).transpose(0,1,2,4,3,5).sum(axis=(4,5))


print(dx.shape)



