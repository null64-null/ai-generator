import numpy as np

def predict(input, layers):
    x = input
    for layer in layers:
        x = layer.func(x)
    return x

def accuracy(input, t, layers):
    prediction = predict(input, layers)

    predicted_indices = np.argmax(prediction, axis=1)
    answer_indices = np.argmax(t, axis=1)

    accuracy = np.sum(predicted_indices == answer_indices)
    
    return accuracy
    


    