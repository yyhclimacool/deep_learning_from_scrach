import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

x = np.arange(-6, 6, 0.01)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, label="cos", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend()
plt.show()

img = imread("./view.png")
plt.imshow(img)
plt.show()
