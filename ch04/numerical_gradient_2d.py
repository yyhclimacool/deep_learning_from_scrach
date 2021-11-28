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

        grad[idx] = (fxh2 - fxh1)/(2*h)
        x[idx] = tmp_val

    return grad

def func(x):
    return x[0]**2 + x[1]**2

print(numerical_gradient_2d(func, np.array([3.0, 4.0])))
print(numerical_gradient_2d(func, np.array([0.0, 2.0])))
print(numerical_gradient_2d(func, np.array([3.0, 0.0])))
