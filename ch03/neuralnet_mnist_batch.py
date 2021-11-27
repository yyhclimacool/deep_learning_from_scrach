#coding: utf-8
#author: yyhclimacool

import numpy as np
import matplotlib.pylab as plt
import pickle
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.functions import *

# 这里不需要使用train数据进行学习，只需要使用标签进行预测即可
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = True, one_hot_label = False)
    return x_test, t_test

# 读取已经学习好的权重参数
def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()
batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    print("Processing: " + str(float(i)/len(x)))
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    t_batch = t[i:i+batch_size]
    accuracy_cnt += np.sum(np.argmax(y_batch, axis=1) == t_batch)

print("Accuracy: " + str(float(accuracy_cnt)/len(x)))
