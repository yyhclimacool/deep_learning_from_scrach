# coding: utf-8

import numpy as np
from PIL import Image
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def img_show(x):
    pil_img = Image.fromarray(np.uint8(x))
    pil_img.show()

(x_train, x_label), (t_train, t_label) = load_mnist(flatten = True, normalize = False)
batch_size = 10
show_mask = np.random.choice(x_train.shape[0], batch_size)
x_show_img = x_train[show_mask]
x_show_label = x_label[show_mask]

for i in range(batch_size):
    img = x_show_img[i]
    label = x_show_label[i]
    print(label)
    img = img.reshape(28, 28)
    img_show(img)

#img = x_train[0]
#label = x_label[0]
#print(label)
#print(img.shape)
#img = img.reshape(28, 28)
#print(img.shape)
#img_show(img)

