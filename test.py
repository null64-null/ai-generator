import os
import numpy as np
import time


# soft max cal test
y = np.array([[1, 2, 1],[1 ,1 ,2]])
dy = np.array([[3, 2, 1],[0 ,2 ,1]])

'''
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
'''

delta = np.eye(y.shape[1])
delta_3d = np.tile(delta, (y.shape[0], 1, 1))
ys_matrix_3d = y[:, np.newaxis, :]
mid_3d = delta_3d - ys_matrix_3d
dx = np.einsum('ij, ijk, ik -> ij', y, mid_3d, dy)
#print(dx)

