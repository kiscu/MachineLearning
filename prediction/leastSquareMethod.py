# coding:utf8
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    """数据导入函数"""
    X = []
    Y = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        X.append(float(curLine[0]))
        Y.append(float(curLine[-1]))
    return X, Y


def plotscatter(Xmat, Ymat, a, b, plt):
    """绘制图形函数"""
    fig = plt.figure()
    ax = fig.add_subplot(111)                      # 绘制图形位置
    ax.scatter(Xmat, Ymat, c='blue', marker='o')   # 绘制散点图
    Xmat.sort()                                    # 对Xmat各元素进行排序
    yhat = [a*float(xi)+b for xi in Xmat]          # 计算预测值
    plt.plot(Xmat, yhat, 'r')                       # 绘制回归线
    plt.title(u"最小二乘法的回归线")
    plt.show()
    return yhat


Xmat, Ymat = loadDataSet("regdataset.txt")         # 导入数据文件
meanX = mean(Xmat)                                 # 原始数据集的均值
meanY = mean(Ymat)
dX = Xmat - meanX                                  # 各元素与均值的差
dY = Ymat - meanY
# 手工计算
# sumXY = 0
# sqX = 0
# for i in xrange(len(dX)):
#     sumXY += double(dX[i])*double(dY[i])
#     SqX += double(dX[i])**2
sumXY = vdot(dX, dY)                               # 返回两个向量的点乘
SqX = sum(power(dX, 2))                            # 向量的平方：（X-meanX）^2

# 计算斜率和截距
a = sumXY/SqX
b = meanY - a*meanX
print a, b                                         # 输出计算后的斜率和截距
# 绘制图形
plotscatter(Xmat, Ymat, a, b, plt)
