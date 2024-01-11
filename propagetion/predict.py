import numpy as np

def predict(input, layers):
    x = input
    for layer in layers:
        x = layer.func(x)
    return x