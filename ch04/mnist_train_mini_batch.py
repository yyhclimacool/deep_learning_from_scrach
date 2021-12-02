#coding:utf-8

import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import *
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# mini-batch的实现
# 分epoch打印精度
def mnist_mini_batch():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    train_size = x_train.shape[0]

    iter_num = 10000
    batch_size = 100
    lr = 0.1

    iter_per_epoch = max(train_size/batch_size, 1) # 60000/100 = 600

    #loss_record_list = []
    train_acc_list = []
    test_acc_list = []
    net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    for i in range(iter_num):
        print("Processing " + str(i) + " of " + str(iter_num))

        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grads = net.numerical_gradient(x_batch, t_batch)
        net.optimize(lr=lr, grads=grads)

        if i % iter_per_epoch == 0:
            #loss_record_list.append(net.loss(x_train, t_train))
            train_acc = net.accuracy(x_train, t_train)
            test_acc = net.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc: " + str(train_acc) + ", test acc: " + str(test_acc))
    for i in range(len(train_acc_list)):
        print("train_acc: " + str(train_acc_list[i]) + ", test_acc: " + str(test_acc_list[i]))

if __name__ == "__main__":
    mnist_mini_batch()
