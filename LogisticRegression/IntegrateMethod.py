# coding:utf8
# Successive approximation method 逐次下降法
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

A = mat([[8, -3, 2], [4, 11, -1], [6, 3, 12]])
b = mat((20, 33, 36))
result = linalg.solve(A, b.T)
print result

error = 1.0e-6
steps = 100
B0 = mat([[0, 3/8, -2/8], [-4/11, 0, 1/11], [-6/12, -3/12, 0]])
f = mat((20/8, 33/11, 36/12))
xk = zeros((3, 1))
errorlist = []
for k in xrange(steps):
    xk_1 = xk
    xk = B0 * xk + f
    errorlist.append(linalg.norm(xk-xk_1))
    if errorlist[-1] < error:
        print k+1
        break
print xk

matpts = zeros((2, k+1))
matpts[0] = linspace(2, k+1, k+1)
matpts[1] = array(errorlist)
plt.scatter(matpts[0], matpts[1])
plt.show()


