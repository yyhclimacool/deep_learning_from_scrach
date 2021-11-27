# coding: utf-8

import numpy as np

# 阶跃函数
def step_function(x):
  return np.array(x > 0, dtype=np.int)

# 用于仿射变换层的激活函数
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# 恒等函数，常用在回归问题上
def identity_function(x):
    return x

# 常用于分类问题的输出层的激活函数
def softmax(x):
  c = np.max(x)
  x = x - c
  exp_a = np.exp(x)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a
  return y
