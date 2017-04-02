# coding:utf8
from numpy import *
import matplotlib.pyplot as plt
import numpy as np


def file2matrix(path, delimiter):
    """
    数据文件转矩阵
    path：数据文件路径
    delimiter：行内字段分隔符
    """
    recordlist = []
    fp = open(path, "rb")                   # 读取文件内容
    content = fp.read()
    fp.close()
    rowlist = content.splitlines()          # 按行转换为一维表
    # 逐行遍历，结果按分隔符分隔为行向量
    recordlist = [map(eval, row.split(delimiter)) for row in rowlist if row.strip()]
    return mat(recordlist)                  # 返回转换后的矩阵形式


def drawScatterbyLabel(plt, Input):
    """按分类绘制散点图"""
    m, n = shape(Input)
    target = Input[:, -1]
    for i in xrange(m):
        if target[i] == 0:
            plt.scatter(Input[i, 0], Input[i, 1], c='blue', marker='o')
        else:
            plt.scatter(Input[i, 0], Input[i, 1], c='red', marker='s')
    # plt.show()


def buildMat(dataSet):
    """构建x+b系数矩阵：b默认为1"""
    m, n = shape(dataSet)
    dataMat = zeros((m, n))
    dataMat[:, 0] = 1                        # 矩阵第一列全为1-->b
    dataMat[:, 1:] = dataSet[:, :-1]         # 第二列到倒数第二列保持原数据，最后一列被删除
    return dataMat


def logistic(wTx):
    """logistic函数"""
    return 1.0/(1.0+exp(-wTx))


def classifier(testData, weights):
    """分类器函数"""
    prob = logistic(sum(testData*weights))  # 求取概率--判别算法
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


# 导入数据
Input = file2matrix("testSet.txt", "\t")    # 导入数据并转换为矩阵
target = Input[:, -1]                       # 获取分类标签列表
[m, n] = shape(Input)

# 按分类绘制散点图
drawScatterbyLabel(plt, Input)

# 构建x+b系数矩阵：b默认为1
dataMat = buildMat(Input)
# print dataMat

alpha = 0.001                               # 步长
steps = 500                                 # 迭代次数
weights = ones((n, 1))                      # 初始化权重向量
weightlist = []

# 执行算法，训练分类器权重
for k in xrange(steps):
    gradient = dataMat * mat(weights)           # 计算梯度
    output = logistic(gradient)                 # logistic函数预测分类
    errors = target - output                    # 计算预测值与实际值之间的误差
    weights = weights + alpha*dataMat.T*errors  # 修正误差，进行迭代
    weightlist.append(weights)

print weights                                   # 输出训练后的权重

# 算法分析

# 超平面的变化趋势
X = np.linspace(-5, 5, 100)
Ylist = []
lenw = len(weightlist)
for indx in xrange(lenw):
    if indx % 20 == 0:                                          # 每隔20条输出一次分类超平面
        weight = weightlist[indx]
        Y = -(double(weight[0])+X*(double(weight[1])))/double(weight[2])
        plt.plot(X, Y)
        plt.annotate("hplane:"+str(indx), xy=(X[99], Y[99]))    # 分类超平面注释
        plt.title(u"分类超平面（权重向量）的变化")
plt.show()

# 超平面的收敛评估

# 1.截距的变化
fig = plt.figure()
axes1 = plt.subplot(211)
axes2 = plt.subplot(212)
weightmat = mat(zeros((steps, n)))
i = 0
for weight in weightlist:
    weightmat[i, :] = weight.T
    i += 1
X = linspace(0, steps, steps)
axes1.plot(X[0:10], -weightmat[0:10, 0]/weightmat[0:10, 2], color='blue', linewidth=1, linestyle="-")
axes2.plot(X[10:], -weightmat[10:, 0]/weightmat[10:, 2], color='blue', linewidth=1, linestyle="-")
plt.title(u"分类超平面曲线的截距变化")
plt.show()

# 2.斜率的变化
fig = plt.figure()
axes1 = plt.subplot(211)
axes2 = plt.subplot(212)
weightmat = mat(zeros((steps, n)))
i = 0
for weight in weightlist:
    weightmat[i, :] = weight.T
    i += 1
X = linspace(0, steps, steps)
axes1.plot(X[0:10], -weightmat[0:10, 1]/weightmat[0:10, 2], color='blue', linewidth=1, linestyle="-")
axes2.plot(X[10:], -weightmat[10:, 1]/weightmat[10:, 2], color='blue', linewidth=1, linestyle="-")
plt.title(u"分类超平面曲线的斜率变化")
plt.show()

# 权重向量的收敛评估
fig = plt.figure()
axes1 = plt.subplot(311)
axes2 = plt.subplot(312)
axes3 = plt.subplot(313)
weightmat = mat(zeros((steps, n)))
i = 0
for weight in weightlist:
    weightmat[i, :] = weight.T
    i += 1
X = linspace(0, steps, steps)
# 输出3个权重分量的变化
axes1.plot(X, weightmat[:, 0], color='blue', linewidth=1, linestyle="-")
axes1.set_ylabel('weight[0]')
axes2.plot(X, weightmat[:, 1], color='red', linewidth=1, linestyle="-")
axes2.set_ylabel('weight[1]')
axes3.plot(X, weightmat[:, 2], color='green', linewidth=1, linestyle="-")
axes3.set_ylabel('weight[2]')
plt.title(u"权重向量的变化趋势")
plt.show()

