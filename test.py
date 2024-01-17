import os
import numpy as np
import time


# soft max cal test
'''
y = np.array([[1, 2, 1],[1 ,1 ,2]])
dy = np.array([[3, 2, 1],[0 ,2 ,1]])

delta = np.eye(y.shape[1])
delta_3d = np.tile(delta[np.newaxis, :, :], (y.shape[0], 1, 1))
y_3d = y.reshape((y.shape[0], 1, y.shape[1]))
y_3d_T = np.transpose(y_3d, axes=(0, 2, 1))
dy_3d = dy.reshape((dy.shape[0], 1, dy.shape[1]))
dy_3d_T = np.transpose(dy_3d, axes=(0, 2, 1))
ys_matrix_3d = np.tile(y[:, np.newaxis, :], (1, y.shape[1], 1))
dxdx_3d = y_3d_T * (delta_3d - ys_matrix_3d)
dx_3d = np.array([ np.dot(dxdx_3d[i], dy_3d_T[i]) for i in range(dy.shape[0])])
dx = dx_3d.reshape((dx_3d.shape[0], dx_3d.shape[1]))
print(dx)

delta = np.eye(y.shape[1])
delta_3d = np.tile(delta, (y.shape[0], 1, 1))
ys_matrix_3d = y[:, np.newaxis, :]
mid_3d = delta_3d - ys_matrix_3d
dx = np.einsum('ij, ijk, ik -> ij', y, mid_3d, dy)
#print(dx)
'''


# convolution test
'''
m = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [2, 1],
    [0, 3]
])

padding = 1
stride = 2

#####

i_len = 1 + (m.shape[0] + 2 * padding - kernel.shape[0]) / stride
j_len = 1 + (m.shape[1] + 2 * padding - kernel.shape[1]) / stride

print(i_len)
print(j_len)

i_len = round(i_len)
j_len = round(j_len)

pad_width_2d = ((padding, padding), (padding, padding)) 
m_pad = np.pad(m, pad_width_2d, mode='constant', constant_values=0)
print(m_pad)

x_next = np.zeros((i_len, j_len))
print(x_next)

for i in range(i_len):
    for j in range(j_len):
        m_divided = m_pad[ i*stride : i*stride + kernel.shape[0], j*stride : j*stride + kernel.shape[1]]
        mk = np.einsum('kl, kl -> kl', m_divided, kernel)
        x_next[i][j] = np.sum(mk)

        print(f"i : {i}")
        print(f"j : {j}")
        print(m_divided)
        print(kernel)
        print(mk)
        print(x_next[i][j])
        print("--------")

print(x_next)
'''