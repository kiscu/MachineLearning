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

# 执行算法，训练分类器权重
for k in xrange(steps):
    gradient = dataMat * mat(weights)           # 计算梯度
    output = logistic(gradient)                 # logistic函数预测分类
    errors = target - output                    # 计算预测值与实际值之间的误差
    weights = weights + alpha*dataMat.T*errors  # 修正误差，进行迭代

print weights                                   # 输出训练后的权重

# 绘制分类超平面
X = np.linspace(-5, 5, 100)
# y=w*x+b: b:weights[0]/weights[2]; w:weights[1]/weights[2]
Y = -(double(weights[0])+X*(double(weights[1])))/double(weights[2])
plt.plot(X, Y)
plt.title(u"权重向量构成的分类超平面")
plt.show()

weights = mat(weights)                          # 权重生成矩阵
testdata = mat([-0.147324, 2.874846])           # 测试数据
m, n = shape(testdata)
testmat = zeros((m, n+1))
testmat[:, 0] = 1
testmat[:, 1:] = testdata
print classifier(testmat, weights)              # 执行分类