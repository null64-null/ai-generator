import sys, os
import numpy as np
import util.graph as graph

from network.initialize import x_init
from network.activation_function import Relu, Sigmoid, Tanh
from network.network import AffineLayer
from propagetion.loss import CrossEntropyError
from propagetion.predict import predict
from propagetion.gradient import generate_grads, update_grads


### mnist ###
sys.path.append(os.pardir)
from mnist.mnist import load_mnist


### input ###
(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True, normalize=True)
train_data_length = x_train.shape[0]

layers = [
    AffineLayer([784, 50]),
    Relu(),
    AffineLayer([50, 10]),
    Sigmoid(),
]

error = CrossEntropyError()
batch_size = 100
lerning_rate = 0.1
epoch = 1000


### main, learning ###
predictions_result = []
errors_result = []

for i in range(epoch):
    batch_mask = np.random.choice(train_data_length, batch_size)
    input = x_train[batch_mask]
    t = t_train[batch_mask]

    prediction = predict(input, layers)
    error.generate_error(prediction, t)
    error.generate_grad(prediction, t)
    generate_grads(layers, error)
    update_grads(layers, lerning_rate)

    predictions_result.append(prediction)
    errors_result.append(error.l)

    print(i)
    print(error.l)
    print("--------")

ai_generated = layers

graph.chart(
    x=list(range(epoch)),
    y=errors_result,
    x_label="iteration",
    y_label="loss",
    title="loss",
)