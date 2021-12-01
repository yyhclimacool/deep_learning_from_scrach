#coding: utf-8

from layers_naive import MulLayer
from layers_naive import AddLayer

apple_price = 100
apple_num = 2
orange_price = 150
orange_num = 3
tax = 1.1

apple_layer = MulLayer()
orange_layer = MulLayer()
apple_orange_layer = AddLayer()
tax_layer = MulLayer()

apple = apple_layer.forward(apple_price, apple_num)
orange = orange_layer.forward(orange_price, orange_num)
total_pay = tax_layer.forward(apple_orange_layer.forward(apple, orange), tax)
print("total pay: " + str(total_pay))

dout = 1
dpre_tax, dtax = tax_layer.backward(dout)
dapple, dorange = apple_orange_layer.backward(dpre_tax)
dapple_price, dapple_num = apple_layer.backward(dapple)
dorange_price, dorange_num = orange_layer.backward(dorange)

print("dapple_price: " + str(dapple_price) + \
        "\n dapple_num: " + str(dapple_num) + \
        "\n dorange_price: " + str(dorange_price) + \
        "\n dorange_num: " + str(dorange_num) + \
        "\n dtax: " + str(dtax))

