import numpy as np

def generate_grads(layers, layer_loss):
    for i in range(len(layers) - 1, -1, -1):
        if i == len(layers) - 1:
            layers[i].generate_grad(layer_loss)
        else:
            layers[i].generate_grad(layers[i+1])

def update_grads(layers, lerning_rate):
    for layer in layers:
        try:
            if(callable(layer.update_grad)):
                layer.update_grad(lerning_rate)
        except:
            pass
        