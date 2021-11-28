#coding:utf-8

import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import *
from dataset.mnist import load_mnist

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        w1, w2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        y1 = sigmoid(np.dot(x, w1) + b1)
        y = softmax(np.dot(y1, w2) + b2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        grad = {}
        f = lambda w: self.loss(x, t)
        grad['W1'] = numerical_gradient(f, self.params['W1'])
        grad['b1'] = numerical_gradient(f, self.params['b1'])
        grad['W2'] = numerical_gradient(f, self.params['W2'])
        grad['b2'] = numerical_gradient(f, self.params['b2'])

        return grad

    def optimize(self, grads, lr=0.01):
        self.params['W1'] -= lr * grads['W1'];
        self.params['b1'] -= lr * grads['b1'];
        self.params['W2'] -= lr * grads['W2'];
        self.params['b2'] -= lr * grads['b2'];

if __name__ == "__main__":
    print("Calling main")
