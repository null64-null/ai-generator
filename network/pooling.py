import numpy as np

class MaxPooling:
    def __init__(self, st):
        self.st = st
        self.x = None
        self.dx = None

        self.dx_filter = None

    def func(self, x):
        n, c, h, w = x.shape
        oh = round(h / self.st)
        ow = round(w / self.st)

        x_reshaped = x.reshape(n, c, oh, self.st, ow, 1, self.st).transpose(0,1,2,4,3,5,6).reshape(n, c, oh, ow, self.st, self.st) 
        x_next = x_reshaped.max(axis=(4, 5)).reshape(n, c, oh, ow)
     
        x_max = x_reshaped.max(axis=(4, 5), keepdims=True)
        x_one_hot = np.where(x_reshaped == x_max, 1, 0)
        
        self.dx_filter = x_one_hot
        self.x = x
        
        return x_next
    
    def generate_grad(self, layer_prev):
        n, c, h, w = self.x.shape
        _, _, oh, ow = layer_prev.dx.shape

        dx_next_reshaped = layer_prev.dx.reshape(n, c, oh, ow, 1, 1)
        dx_next_filtered = self.dx_filter * dx_next_reshaped
        self.dx = dx_next_filtered.transpose(0,1,2,4,3,5).reshape(n, c, w, h)


class AveragePooling:
    def __init__(self, st):
        self.st = st
        self.x = None
        self.dx = None

        self.dx_filter = None

    def func(self, x):
        n, c, h, w = x.shape
        oh = round(h / self.st)
        ow = round(w / self.st)

        x_reshaped = x.reshape(n, c, oh, self.st, ow, 1, self.st).transpose(0,1,2,4,3,5,6).reshape(n, c, oh, ow, self.st, self.st) 
        x_next = (x_reshaped.sum(axis=(4, 5)) / (self.st * self.st)).reshape(n, c, oh, ow)
        
        self.dx_filter = np.ones(x_reshaped.shape)

        self.x = x
        return x_next
    
    def generate_grad(self, layer_prev):
        n, c, h, w = self.x.shape
        _, _, oh, ow = layer_prev.dx.shape

        dx_next_reshaped = layer_prev.dx.reshape(n, c, oh, ow, 1, 1)
        dx_next_filtered = self.dx_filter * dx_next_reshaped
        self.dx = dx_next_filtered.transpose(0,1,2,4,3,5).reshape(n, c, w, h)
        
        
        