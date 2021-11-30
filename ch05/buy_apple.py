#coding: utf-8

from layers_naive import MulLayer

apple_price = 100
apple_num = 2
tax = 1.1

layer1 = MulLayer()
apple_pay = layer1.forward(apple_price, apple_num)
layer2 = MulLayer()
total_pay = layer2.forward(apple_pay, tax)
print("total pay: " + str(total_pay) + " (200 * 2 * 1.1)")

dout = 1
dapple_pay, dtax = layer2.backward(dout)
dapple_price, dapple_num = layer1.backward(dapple_pay)
print("dapple_price: " + str(dapple_price) + ", dapple_num: " + str(dapple_num) + ", dtax: " + str(dtax))
