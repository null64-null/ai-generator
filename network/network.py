import numpy as np
from .initialize import w_init, b_init, c_init

# network for affine
class AffineLayer:
    def __init__(self, layer_sizes, weight_init_std=0.01):
        self.x = None
        self.w = weight_init_std * np.random.randn(layer_sizes[0], layer_sizes[1])
        self.b = np.zeros(layer_sizes[1])
        self.dx = None
        self.dw = None
        self.db = None
        
    def func(self, x):
        self.x = x
        x_next = np.dot(self.x, self.w) + self.b
        return  x_next

    def generate_grad(self, layer_prev):
        self.dw = np.dot(self.x.T, layer_prev.dx)
        self.db = np.sum(layer_prev.dx, axis=0)
        self.dx = np.dot(layer_prev.dx, self.w.T)
    
    def update_grad(self, lerning_rate):
        self.w -= self.dw * lerning_rate
        self.b -= self.db * lerning_rate

# network for conb