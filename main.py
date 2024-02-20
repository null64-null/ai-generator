import sys, os
import numpy as np

from util.graph import chart
from util.progress import show_iter_progress

from network.activation import Relu, Sigmoid, Tanh, Softmax
from network.network import AffineLayer, ConvolutionLayer, FlattenSection
from propagetion.loss import CrossEntropyError
from propagetion.predict import predict, accuracy
from propagetion.gradient import generate_grads, update_grads
from util.picture import row2im

# mnist 
sys.path.append(os.pardir)
from mnist.mnist import load_mnist


### input ###
#input
# make input test　(x_train, t_train), (x_test, t_test), image shape
(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True, normalize=True)
train_data_length = x_train.shape[0]
test_data_length = x_test.shape[0]
x_train = x_train.reshape(train_data_length, 1, 28, 28)
x_test = x_test.reshape(test_data_length, 1, 28, 28)

batch_size = 100 #input


### network ###
error = CrossEntropyError()  #input
lerning_rate = 0.1  #input

# layers
# st = 1, pad = 0
# oh = 1 + (h + 2 * pad - fh) / st
# ow = 1 + (w + 2 * pad - fw) / st
layers = [
    # (n, c, h, w) = (batch_size, 1, 28, 28)

    ConvolutionLayer(
        # input_size=[batch_size, 1, 28, 28],
        filter_size=[5, 1, 6, 6],
        pad = 0,
        st = 1
    ),
    Relu(),
    # (n, fn, oh, ow) = (batch_size, 5, 23, 23)

    ConvolutionLayer(
        # input_size=[batch_size, 5, 23, 23],
        filter_size=[10, 5, 12, 12],
        pad = 0,
        st = 1
    ),
    Relu(),
    # (n, fn, oh, ow) = (batch_size, 10, 12, 12)

    FlattenSection(), 
    # (h, w) = (batch_size, 12*12*10) = (batch_size, 1440)

    AffineLayer([1440, 10]),
    Softmax(),
] #input

### times learning, accuracy check span  ###
iters = 10000 #input
accuracy_check_span = 100 #input
iter_per_epoch = max(train_data_length / batch_size, 1) # Howmany iters per epoch 


# main, learning 
predictions_result = []
errors_result = []
accuracy_result_train_data = []
accuracy_result_test_data = []
accuracy_x_axis = []

for i in range(iters):
    # choose train data from data set
    batch_mask = np.random.choice(train_data_length, batch_size)
    input = x_train[batch_mask]
    t = t_train[batch_mask]

    # do prediction
    prediction = predict(input, layers)

    # calculate error using prediction result
    error.generate_error(prediction, t)

    # calculate gradients
    error.generate_grad(prediction, t)
    generate_grads(layers, error)

    # update parameters in layers
    update_grads(layers, lerning_rate)


    '''
    # if 1 epoch done, check accuracy 
    # if i % iter_per_epoch == 0:
    if i % accuracy_check_span == 0:
        test_layers = layers
        train_data_accuracy = accuracy(x_train, t_train, test_layers)
        test_data_accuracy = accuracy(x_test, t_test, test_layers)
        
        # recode results (accuracy)
        accuracy_result_train_data.append(train_data_accuracy)
        accuracy_result_test_data.append(test_data_accuracy)
        accuracy_x_axis.append(i+1)
        
    '''
    # recode results (prediction, error)
    predictions_result.append(prediction)
    errors_result.append(error.l)

    # indicator
    show_iter_progress(
        i+1,
        iters,
        error.l,
        accuracy_check_span,
        train_data_length,
        test_data_length,
        accuracy_result_train_data,
        accuracy_result_test_data,
        accuracy_x_axis,
    )

ai_generated = layers

chart(
    x=list(range(iters)),
    y=errors_result,
    x_label="iteration",
    y_label="loss",
    title="loss",
)

chart(
    x=accuracy_x_axis,
    y=accuracy_result_test_data,
    x_label="iteration",
    y_label="accuracy",
    title="accuracy (test)",
)