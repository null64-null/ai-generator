import numpy as np

class Relu:
    def __init__(self):
        self.x = None
        self.dx = None

    def func(self, x):
        self.x = x
        x_next = np.maximum(0, self.x)
        return x_next
    
    def generate_grad(self, layer_prev):
       dxdx = self.x > 0
       self.dx = dxdx * layer_prev.dx  

class Sigmoid:
    def __init__(self):
        self.x = None
        self.dx = None

    def func(self, x):
        self.x = x
        next_x = 1 / (1 + np.exp(-self.x))
        return next_x
    
    def generate_grad(self, layer_prev):
        dxdx = layer_prev.x * (1 - layer_prev.x)
        self.dx = dxdx * layer_prev.dx

class Tanh:
    def __init__(self):
        self.x = None
        self.dx = None

    def func(self, x):
        self.x = x
        next_x = np.tanh(self.x)
        return next_x
    
    # 要確認
    def generate_grad(self, layer_prev):
        dxdx = 1 - np.power(layer_prev.dx, 2)
        self.dx = dxdx * layer_prev.dx

class Softmax:
    def __init__(self):
        self.x = None
        self.dx = None

    def func(self, x):
        self.x = x
        x = x - np.max(x, axis=-1, keepdims=True)
        next_x = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
        return next_x
    
    def generate_grad(self, layer_prev):
        delta = np.eye(layer_prev.x.shape[1])
        delta_3d = np.tile(delta, (layer_prev.x.shape[0], 1, 1))
        ys_matrix_3d = layer_prev.x[:, np.newaxis, :]
        mid_3d = delta_3d - ys_matrix_3d
        self.dx = np.einsum('ij, ijk, ik -> ij', layer_prev.x, mid_3d, layer_prev.dx)