# coding: utf-8

import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

batch_mask = np.random.choice(x_train.shape[0], 10)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

nume_grad = network.numerical_gradient(x_batch, t_batch)
back_grad = network.gradient(x_batch, t_batch)

# TODO: diff is too big ...
for key in nume_grad.keys():
    diff = np.average(np.abs(back_grad[key] - nume_grad[key]))
    print(key + " : " + str(diff))
