import numpy as np

def im2col(input, fh, fw, oh, ow, pad, st):
    n, c, _, _ = input.shape

    col = np.zeros((n, c, fh*fw, oh*ow))
    imput_pad = np.pad(input, ((0, 0), (0, 0), (pad, pad),(pad, pad)), mode='constant', constant_values=0)

    for i in range(fh):
        for j in range(fw):
            col[:, :, i*fw+j:i*fw+j+1, :] = imput_pad[:, :, i:oh*st+i:st, j:ow*st+j:st].reshape((n, c, 1, oh*ow))

    return col

def col2im(col, n, c, fh, fw, h, w, oh, ow, pad, st):
    input = np.zeros((n, c, h, w))
    input_pad = np.zeros((n, c, h+pad*2, w+pad*2))

    for i in range(fh):
        for j in range(fw):
            input_pad[:, :, i:oh*st+i:st, j:ow*st+j:st] += col[:, :, i*fw+j:i*fw+j+1, :].reshape((n, c, ow, oh))

    if pad != 0:
        input = input_pad[:, :, 1:-pad, 1:-pad]
    
    input = input_pad

    return input

def im2col_deconv(output, n, fn, fh, fw, h, w, st):
    col = np.zeros((n, fn, fh*fw, h*w))

    for i in range(fh):
        for j in range(fw):
            col[:, :, i*fw+j:i*fw+j+1, :] = output[:, :, i:h*st+i:st, j:w*st+j:st].reshape((n, fn, 1, h*w))

    return col

def col2im_deconv(col, n, fn, fh, fw, h, w, oh, ow, st):
    output_pad = np.zeros((n, fn, oh, ow))

    for i in range(fh):
        for j in range(fw):
            output_pad[:, :, i:h*st+i:st, j:w*st+j:st] += col[:, :, i*fw+j:i*fw+j+1, :].reshape((n, fn, h, w))

    return output_pad

def compress_z(z, n, fn, oh, ow):
    return z.reshape(n, fn, 1, oh*ow).transpose(1, 0, 2, 3).reshape(fn, oh*ow*n).T

def deploy_z(z, n, fn, oh, ow):
    return z.T.reshape(fn, n, 1, oh*ow).transpose(1, 0, 2, 3).reshape(n, fn, oh, ow)

def compress_w(w, fn, c, fh, fw):
    return w.reshape(fn, fh*fw*c).T

def deploy_w(w, fn, c, fh, fw):
    return w.T.reshape(fn, c, fh, fw)

def compress_xcol(x, n, c, h, w):
    return x.transpose(0, 2, 1, 3).reshape(n*h, c*w)

def deploy_xcol(x, n, c, h, w):
    return x.reshape(n, h, c, w).transpose(0, 2, 1, 3)