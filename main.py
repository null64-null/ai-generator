import sys, os
import numpy as np
import matplotlib.pyplot as plt

from util.graph import show_graphs, show_results
from util.progress import show_iter_progress

from network.activation import Relu, Sigmoid, Tanh, Softmax
from network.network import AffineLayer, ConvolutionLayer, FlattenSection
from propagetion.loss import CrossEntropyError
from propagetion.predict import predict, accuracy, calculate_accuracy
from propagetion.gradient import generate_grads, update_grads

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
iters = 100 #input
accuracy_check_span = 50 #input
check_mask_size_train = 50 #input
check_mask_size_test = 50 #input
iter_per_epoch = max(train_data_length / batch_size, 1) # Howmany iters per epoch 


# main, learning 
errors_x_axis = []
errors = []
accuracy_x_axis = []
accuracy_ratios_train = []
accuracy_ratios_test = []

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

    # recode results (error), update graph
    errors.append(error.l)
    errors_x_axis.append(i)

    # update parameters in layers
    update_grads(layers, lerning_rate)

    # accuracy check
    if i % accuracy_check_span == 0:
        # get contemporary layer
        test_layers = layers

        # get accuracy from train, test data
        accurate_predictions_train, all_data_train, accuracy_ratio_train = calculate_accuracy(x_train, t_train, test_layers, check_mask_size_train)
        accurate_predictions_test, all_data_test, accuracy_ratio_test = calculate_accuracy(x_test, t_test, test_layers, check_mask_size_test)
        
        # recode results (accuracy), update graph
        accuracy_ratios_train.append(accuracy_ratio_train)
        accuracy_ratios_test.append(accuracy_ratio_test)
        accuracy_x_axis.append(1+i*accuracy_check_span)
        
    # indicator iter
    show_iter_progress(
        i+1,
        iters,
        error.l,
        accuracy_check_span,
        accurate_predictions_train,
        all_data_train,
        accuracy_ratio_train,
        accurate_predictions_test,
        all_data_test,
        accuracy_ratio_test,
        accuracy_x_axis,
    )

    # show graph
    show_graphs(
        x1=errors_x_axis,
        y1=errors,
        x2=accuracy_x_axis,
        y2_1=accuracy_ratios_train,
        y2_2=accuracy_ratios_test,
        x1_label='iter [times]',
        y1_label='error [-]',
        x2_label='iter [times]',
        y2_label='accuracy [%]',
        y1_name='error',
        y2_1_name='accuracy (train)',
        y2_2_name='accuracy (test)',
        title1='error',
        title2='accuracy',
    )

# show graph
show_results(
    x1=errors_x_axis,
    y1=errors,
    x2=accuracy_x_axis,
    y2_1=accuracy_ratios_train,
    y2_2=accuracy_ratios_test,
    x1_label='iter [times]',
    y1_label='error [-]',
    x2_label='iter [times]',
    y2_label='accuracy [%]',
    y1_name='error',
    y2_1_name='accuracy (train)',
    y2_2_name='accuracy (test)',
    title1='error',
    title2='accuracy',
)

