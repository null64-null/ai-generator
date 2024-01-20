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
class ConvolutionLayer:
    def __init__(self, layer_sizes, padding, stride, weight_init_std=0.01):
        self.x = None
        self.w = weight_init_std * np.random.randn(layer_sizes[0], layer_sizes[1])
        self.dw = None
        self.dx = None
        self.padding = padding
        self.stride = stride

    def func(self, x):
        self.x = x

        i_len = 1 + (self.x.shape[0] + 2 * self.padding - self.w.shape[0]) / self.stride
        j_len = 1 + (self.x.shape[1] + 2 * self.padding - self.w.shape[1]) / self.stride
        i_len = round(i_len)
        j_len = round(j_len)

        x_next = np.zeros((i_len, j_len))

        x_pad_width = ((self.padding, self.padding), (self.padding, self.padding)) 
        x_pad = np.pad(self.x, x_pad_width, mode='constant', constant_values=0)

        for i in range(i_len):
            for j in range(j_len):
                x_divided = x_pad[ i*self.stride : i*self.stride + self.w.shape[0], j*self.stride : j*self.stride + self.w.shape[1]]
                xw = np.einsum('kl, kl -> kl', x_divided, self.w)
                x_next[i][j] = np.sum(xw)
        
        return x_next
    
    def generate_grad(self, layer_prev):
        # get dx
        self.dx = np.zeros(self.x.shape)
        dxdx = np.array([])

        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                dxdx_i = np.array([])

                for yi in range(self.x.shape[0]):
                    for yj in range(self.x.shape[1]):
                        wi_index = -1
                        wj_index = -1

                        for wi in range(self.w.shape[0]):
                            for wj in range(self.w.shape[1]):
                                if(i == yi * self.stride + wi):
                                    wi_index = i - yi * self.stride            
                                if(j == yj * self.stride + wj):
                                    wj_index = j - yj * self.stride

                        if (wi_index == -1 &  wj_index == -1):
                            dxdx_i = np.append(0, dxdx_i)
                        else:
                            dxdx_i = np.append(self.w[wi_index][wj_index], dxdx_i)
            
                dxdx = np.append(dxdx, dxdx_i, axis=0)
        
        dx_1d = np.dot(dxdx, layer_prev.dx.flatten())
        self.dx = np.reshape(dx_1d, self.x.shape)

        # get dw
        self.dw = np.zeros(self.w.shape)
        dxdw = np.array([])

        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                dxdw_i = np.array([])

                for yi in range(self.x.shape[0]):
                    for yj in range(self.x.shape[1]):
                        dxdw_i = np.append(self.x[yi * self.stride + i][yj * self.stride + j], dxdw_i)
                
                dxdw = np.append(dxdw, dxdw_i, axis=0)

        dw_1d = np.dot(dxdw, layer_prev.dx.flatten())
        self.dw = np.reshape(dw_1d, self.w.shape)

    def update_grad(self, lerning_rate):
        self.w -= self.dw * lerning_rate