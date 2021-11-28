# coding: utf-8

import numpy as np

def numerical_gradient_2d(f, x):
    h = 1e-5
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2)/(2*h)
        x[idx] = tmp_val

    return grad

def func(x):
    return x[0]**2 + x[1]**2

def gradient_descent(f, init_x, lr = 0.1, iter_num = 100):
    x = init_x
    for i in range(iter_num):
        grad = numerical_gradient_2d(f, x)
        x -= lr * grad
    return x

init_x = np.array([-3.0, 4.0])
result = gradient_descent(f=func, init_x=init_x, lr=0.1)
print(result)
