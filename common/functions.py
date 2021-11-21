# coding: utf-8

import numpy as np

def step_function(x):
  return np.array(x > 0, dtype=np.int)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identity_function(x):
    return x

def softmax(x):
  c = np.max(x)
  x = x - c
  exp_a = np.exp(x)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  return y
