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
    
def calculate_accuracy(input, t, layers, divided_size):
    iters = round(input.shape[0] / divided_size)
    acc = 0
    for i in range(iters):
        masked_input = input[i*divided_size : (i+1)*divided_size]
        masked_t = t[i*divided_size : (i+1)*divided_size]
        acc += accuracy(masked_input, masked_t, layers)
        print(f"accurate calcuration progress ...  : {i} / {iters} done", end="\r")

    accurate_predictions = acc
    all_data = input.shape[0]
    accuracy_ratio = (acc / all_data) * 100
    
    return accurate_predictions, all_data, accuracy_ratio
    