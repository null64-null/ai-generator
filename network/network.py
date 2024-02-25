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
    def __init__(self, filter_size, pad, st, weight_init_std=0.01):
        self.filter_size = filter_size
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
        n, c, h, w = x.shape
        fn, c, fh, fw = self.filter_size

        oh = round(1 + (h + 2 * self.pad - fh) / self.st)
        ow = round(1 + (w + 2 * self.pad - fw) / self.st)

        # colize and compress x
        x_col = im2col(x, fh, fw, oh, ow, self.pad, self.st)
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
        n, c, h, w = self.x.shape
        fn, c, fh, fw = self.filter_size

        dx_next = layer_prev.dx

        dz = dx_next
        dz_com = compress_z(dz, n, fn, oh, ow)

        dx_com = np.dot(dz_com, self.w_com.T)        
        dx_col_T = deploy_xcol(dx_com, n, c, oh*ow, fh*fw)
        dx_col = dx_col_T.transpose(0, 1, 3, 2)
        dx = col2im(dx_col, fh, fw, h, w, oh, ow, self.pad, self.st)
        self.dx = dx
        
        dw_com = np.dot(self.x_com.T, dz_com)
        dw = deploy_w(dw_com, fn, c, fh, fw)
        self.dw = dw

        db = np.sum(dx_next, axis=(0, 2, 3))
        self.db = db

    def update_grad(self, lerning_rate):
        self.w -= self.dw * lerning_rate
        self.b -= self.db * lerning_rate


# network for de-conb
class DeconvolutionLayer:
    def __init__(self, filter_size, pad, st, weight_init_std=0.01):
        self.filter_size = filter_size
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
        n, c, w, h = x.shape
        fn, c, fh, fw = self.filter_size

        oh = (h - 1)*self.st - self.pad*2 + fh
        ow = (h - 1)*self.st - self.pad*2 + fw

        # compress x
        x_com = x.reshape(n, c, 1, h*w).transpose(1, 0, 2, 3).reshape(c, n*h*w)

        # compress w
        w_com = w.reshape(fn, c, fh*fw).transpose(0, 2, 1).reshape(fn*fh*fw, c)

        # convolute
        z_com = np.dot(w_com, x_com)

        # deploy z
        z_col = z_com.reshape(c, n, fh*fw, h*w).transpose(0,2,1,3).transpose(1, 0, 2, 3)
        z = col2im(z_col, fh, fw, h, w, oh, ow, self.pad, self.st) #要確認

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
        n, c, h, w = self.x.shape
        fn, c, fh, fw = self.filter_size

        dx_next = layer_prev.dx

        dz = dx_next
        dz_col = im2col(dz, fh, fw, oh, ow, self.pad, self.st)
        dz_com = dz_col.transpose(1, 0, 2, 3).transpose(0,2,1,3).reshape(fn*fh*fw, h*w*n)

        dw_com = np.dot(dz_com, self.x_com.T)
        dw = dw_com.reshape(fn, fh*fw, c).transpose(0, 2, 1).reshape(fn, c, fh, fw)
        self.dw = dw

        dx_com = np.dot(self.w_com.T, dz_com)
        dx = dx_com.T.reshape(n, h*w, c).transpose(0, 2, 1).reshape(n, c, h, w)
        self.dx = dx

        db = np.sum(dx_next, axis=(0, 2, 3))
        self.db = db

    def update_grad(self, lerning_rate):
        self.w -= self.dw * lerning_rate
        self.b -= self.db * lerning_rate


# joint convolution and affine
class FlattenSection:
    def __init__(self):
        self.x = None
        self.dx = None

    def func(self, x):
        n, _, _, _ = x.shape
        
        x_next = x.reshape(n, -1)

        self.x = x
        return x_next
    
    def generate_grad(self, layer_prev):
        dx_next = layer_prev.dx #flat
        self.dx = dx_next.reshape(self.x.shape) #vertical
        

# joint convolution and affine
class VerticalizeSection:
    def __init__(self):
        self.x = None
        self.dx = None

    def func(self, x, next_x_shape):
        n, chw = x.shape
        _, c, oh, ow = next_x_shape
        
        x_next = x.reshape(n, c, oh, ow)

        self.x = x
        return x_next
    
    def generate_grad(self, layer_prev):
        dx_next = layer_prev.dx #vertical
        self.dx = dx_next.reshape(self.x.shape) #flat


