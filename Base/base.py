# coding:utf8
import numpy as np
from numpy import *

# 1.矩阵的初始化
myZero = np.zeros([3, 5])
print myZero
myOnes = np.ones([3, 5])
print myOnes
myRand = np.random.rand(3, 4)
print myRand
myEye = np.eye(3)
print myEye

# 2.矩阵的元素运算
myOnes = ones([3, 3])
myEye = eye(3)
print myOnes+myEye
print myOnes-myEye

mymatrix = mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
a = 10
print a * mymatrix
print sum(mymatrix)

mymatrix2 = 1.5 * ones([3, 3])
# 矩阵的点乘同维对应元素的相乘
print multiply(mymatrix, mymatrix2)

# 矩阵各元素的n次幂
print power(mymatrix, 2)
mylist = mat([1, 2, 3])
print power(mylist, 2)