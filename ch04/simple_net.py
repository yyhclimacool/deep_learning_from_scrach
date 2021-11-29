#coding: utf-8

import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import *

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
        print("SimpleNet.self.W:")
        print(self.W)

    def predict(self, x):
        y = np.dot(x, self.W)
        #print("SimpleNet.predict result:")
        #print(y)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        z = softmax(y)
        error = cross_entropy_error(z, t)
        #print("SimpleNet.loss result:")
        #print(error)
        return error

    def numerical_gradient(self, x, t):
        f = lambda W: self.loss(x, t)
        grad = numerical_gradient(f, self.W)
        print("grad: ")
        print(grad)
        return grad

simple_net = SimpleNet()
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])
simple_net.numerical_gradient(x, t)
