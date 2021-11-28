#coding:utf-8

import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import *
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# mini-batch的实现
def mnist_mini_batch():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    for i in range(x_train.shape[0]):
        print("Processing " + str(i) + " of " + str(x_train.shape[0]))
        grads = net.numerical_gradient(x_train[i], t_train[i])
        net.optimize(lr=0.01, grads=grads)
        if i != 0 and i % 100 == 0:
            print(net.accuracy(x_test[i-100:i], t_test[i-100:i]))

mnist_non_mini_batch()
