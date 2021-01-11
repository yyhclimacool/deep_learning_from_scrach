#!/bin/env python3
import numpy as np

def softmax(x):
  c = np.max(x)
  x = x - c
  exp_a = np.exp(x)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  return y

x = np.array([0.3, 2.9, 4.0])
print(softmax(x))
