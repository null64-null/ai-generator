def row2im(x, n, c, h, w):
    return x.reshape(n, c, h, w)

def im2row(x, h, w):
    return x.reshape(-1, h*w)