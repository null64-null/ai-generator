import numpy as np
from network.functions.convolution_functions import im2col, col2im, compress_xcol, compress_w, compress_z, deploy_xcol, deploy_w, deploy_z

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
class ConvolutionLayer:
    def __init__(self, input_size, filter_sizes, pad, st, weight_init_std=0.01):        
        self.input_size = input_size #(n, c, h, w)
        self.filter_size = filter_sizes #(fn, c, fh, fw)
        n, c, h, w = self.input_size
        fn, c, fh, fw = self.filter_size

        self.pad = pad
        self.st = st
    
        self.x = None 
        self.w = weight_init_std * np.random.randn(fn, c, fh, fw)
        self.b = weight_init_std * np.random.randn(fn)
        
        self.dx = None
        self.dw = None
        self.db = None

        self.x_com = None
        self.w_com = None

    def func(self, x):
        n, c, h, w = self.input_size
        fn, c, fh, fw = self.filter_size

        oh = round(1 + (h + 2 * self.pad - fh) / self.st)
        ow = round(1 + (w + 2 * self.pad - fw) / self.st)

        # colize and compress x
        x_col = im2col(x, fh, fw, self.pad, self.st)
        x_col_T = x_col.transpose(0, 1, 3, 2)
        x_com = compress_xcol(x_col_T, n, c, oh*ow, fh*fw)
        
        # compress w
        w_com = compress_w(self.w, fn, c, fh, fw)
       
        # convolute
        z_com = np.dot(x_com, w_com)

        # deploy z
        z = deploy_z(z_com, n, fn, oh, ow)
    
        # verticalize b
        b_ver = self.b.reshape(fn, 1, 1)

        # add bias
        x_next = z + b_ver

        # save params
        self.x = x
        self.x_com = x_com
        self.w_com = w_com

        return x_next
    
    def generate_grad(self, layer_prev):
        n, fn, oh, ow = layer_prev.x.shape
        n, c, h, w = self.input_size
        fn, c, fh, fw = self.filter_size

        dx_next = layer_prev.dx

        dz = dx_next
        dz_com = compress_z(dz, n, fn, oh, ow)

        dx_com = np.dot(dz_com, self.w_com.T)        
        dx_col_T = deploy_xcol(dx_com, n, c, oh*ow, fh*fw)
        dx_col = dx_col_T.transpose(0, 1, 3, 2)
        dx = col2im(dx_col, fh, fw, oh, ow, self.pad, self.st)
        self.dx = dx
        
        dw_com = np.dot(self.x_com.T, dz_com)
        dw = deploy_w(dw_com, fn, c, fh, fw)
        self.dw = dw

        db = np.sum(dx_next, axis=(0, 2, 3))
        self.db = db

    def update_grad(self, lerning_rate):
        self.w -= self.dw * lerning_rate
        self.b -= self.db * lerning_rate