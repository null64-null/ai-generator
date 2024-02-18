import os
import numpy as np
import time
from network.network import ConvolutionLayer

# conv y, dy reshape
'''
n = 2
oh = 2
ow = 3
fn = 3

dy2 = np.array([
    [11,21,31],
    [12,22,32],
    [13,23,33],
    [14,24,34],
    [15,25,35],
    [16,26,36],
    [41,51,61],
    [42,52,62],
    [43,53,63],
    [44,54,64],
    [45,55,65],
    [46,56,66],
])

a = dy2.T
b = a.reshape(fn, n, 1, oh*ow)
c = b.transpose(1, 0, 2, 3)
d = c.reshape(n, fn, oh, ow)

dy = np.array([
    [
        [
            [11,12,13],
            [14,15,16],
        ],
        [
            [21,22,23],
            [24,25,26],
        ],
          [
            [31,32,33],
            [34,35,36],
        ],
    ],
    [
        [
            [41,42,43],
            [44,45,46],
        ],
        [
            [51,52,53],
            [54,55,56],
        ],
          [
            [61,62,63],
            [64,65,66],
        ],
    ],
])

a1 = dy.reshape(n, fn, 1, oh*ow)
b1 = a1.transpose(1, 0, 2, 3)
c1 = b1.reshape(fn, oh*ow*n)
d1 = c1.T

def compress_y(y, n, fn, oh, ow):
    return y.reshape(n, fn, 1, oh*ow).transpose(1, 0, 2, 3).reshape(fn, oh*ow*n).T

def deploy_y(y, n, fn, oh, ow):
    return y.T.reshape(fn, n, 1, oh*ow).transpose(1, 0, 2, 3).reshape(n, fn, oh, ow)
'''


# conv w, dw reshape
'''
c = 2
fn = 3
fh = 2
fw = 3

dw2 = np.array([
    [11,31,51],
    [12,32,52],
    [13,33,53],
    [14,34,54],
    [15,35,55],
    [16,36,56],
    [21,41,61],
    [22,42,62],
    [23,43,63],
    [24,44,64],
    [25,45,65],
    [26,46,66],
])

a = dw2.T
b = a.reshape(fn, c, fh, fw)

dw = np.array([
    [
        [
            [11,12,13],
            [14,15,16],
        ],
        [
            [21,22,23],
            [24,25,26],
        ],     
    ],
    [
        [
            [31,32,33],
            [34,35,36],
        ],
        [
            [41,42,43],
            [44,45,46],
        ],
    ],
    [
        [
            [51,52,53],
            [54,55,56],
        ],
        [
            [61,62,63],
            [64,65,66],
        ],
    ]
])

a1 = dw.reshape(fn, fh*fw*c)
b1 = a1.T

def compress_w(w, fn, c, fh, fw):
    return w.reshape(fn, fh*fw*c).T

def deploy_w(w, fn, c, fh, fw):
    return w.T.reshape(fn, c, fh, fw)
'''

# conv x col T, dx col T reshape
'''
n = 2
c = 3
h = 2
w = 3

dx = np.array([
    [11, 12, 13, 21, 22, 23, 31, 32, 33],
    [14, 15, 16, 24, 25, 26, 34, 35, 36],
    [41, 42, 43, 51, 52, 53, 61, 62, 63],
    [44, 45, 46, 54, 55, 56, 64, 65, 66],
])

reverse = dx.reshape(n, h, c, w).transpose(0, 2, 1, 3)

x = np.array([
    [
        [
            [11, 12, 13],
            [14, 15, 16]
        ],
        [
            [21, 22, 23],
            [24, 25, 26]
        ],
        [
            [31, 32, 33],
            [34, 35, 36],
        ],
    ],
    [
        [
            [41, 42, 43],
            [44, 45, 46]
        ],
        [
            [51, 52, 53],
            [54, 55, 56]
        ],
        [
            [61, 62, 63],
            [64, 65, 66]
        ],
    ],
])

com = x.transpose(0, 2, 1, 3).reshape(n*h, c*w)

def compress_xcol(x, n, c, h, w):
    return x.transpose(0, 2, 1, 3).reshape(n*h, c*w)

def deploy_xcol(x, n, c, h, w):
    return x.reshape(n, h, c, w).transpose(0, 2, 1, 3)
'''



# b, db
'''
dy = np.array([
    [
        [
            [11,12,13],
            [14,15,16],
        ],
        [
            [21,22,23],
            [24,25,26],
        ],
          [
            [31,32,33],
            [34,35,36],
        ],
    ],
    [
        [
            [41,42,43],
            [44,45,46],
        ],
        [
            [51,52,53],
            [54,55,56],
        ],
          [
            [61,62,63],
            [64,65,66],
        ],
    ],
])

b = np.array([1,2,3])
c = b.reshape(3,1,1)
print(c)
print(dy + c)
'''

n = 2
c = 3
h = 3
w = 3

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
          [
            [31,32,33],
            [34,35,36],
            [37,38,39],
        ],
    ],
    [
        [
            [41,42,43],
            [44,45,46],
            [47,48,49],
        ],
        [
            [51,52,53],
            [54,55,56],
            [57,58,59],
        ],
          [
            [61,62,63],
            [64,65,66],
            [67,68,69],
        ],
    ],
])

fn = 3
c = 3
fh = 2
fw = 2

f = np.array([
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
    [
        [
            [71,72],
            [73,74],
        ],
        [
            [81,82],
            [83,84],
        ],
        [
            [91,92],
            [93,94],
        ],
    ],
])

a = np.array([
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

class NextLayer:
    def __init__(self, x_next):
        self.x = x_next
        self.dx = a

layer = ConvolutionLayer([n,c,h,w], [fn,c,fh,fw], 0, 1)

x_next = layer.func(x)
layer_prev = NextLayer(x_next)

layer.generate_grad(layer_prev)

print(layer.dx)





