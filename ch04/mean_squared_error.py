#coding: utf-8

import numpy as np

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])

print(t)
print(y)
print(mean_squared_error(y, t))
