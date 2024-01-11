import numpy as np

class CrossEntropyError:
    def __init__(self):
        self.x = None
        self.dx = None
        self.l = None
    
    def generate_error(self, x, t):
        self.x = x
        if self.x.ndim == 1:
                t = t.reshape(1, t.size)
                self.x = self.x.reshape(1, self.x.size)

        batch_size = self.x.shape[0]
        self.l = -np.sum(t * np.log(self.x + 1e-7)) / batch_size
    
    def generate_grad(self, x, t):
        batch_size = x.shape[0]
        self.dx = -(t / (x + 1e-7)) / batch_size
    