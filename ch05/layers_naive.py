#coding: utf-8

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y

        return x * y

    def backward(self, dout):
        dx = self.y * dout
        dy = self.x * dout

        return dx, dy
