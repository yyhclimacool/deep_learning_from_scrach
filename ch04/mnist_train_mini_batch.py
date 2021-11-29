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
    iter_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    loss_record_list = []
    net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    for i in range(iter_num):
        mini_batch = np.random.choice(train_size, batch_size)
        x_batch = x_train[mini_batch]
        t_batch = t_train[mini_batch]
        print("Processing " + str(i) + " of " + str(iter_num))
        grads = net.numerical_gradient(x_batch, t_batch)
        net.optimize(lr=0.1, grads=grads)
        loss_record_list.append(net.loss(x_batch, t_batch))
    for loss in loss_record_list:
        print(loss)

if __name__ == "__main__":
    mnist_mini_batch()
