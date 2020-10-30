#!/bin/evn python

from and_gate import AND
from nand_gate import NAND
from or_gate import OR

def XOR(x1, x2):
  a = NAND(x1, x2)
  b = OR(x1, x2)
  return AND(a, b)

if __name__ == "__main__":
  print XOR(0, 0)
  print XOR(0, 1)
  print XOR(1, 0)
  print XOR(1, 1)
