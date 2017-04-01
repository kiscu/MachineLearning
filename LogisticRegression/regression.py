# coding:utf8
from numpy import *
import matplotlib.pyplot as plt


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
    plt.show()


def buildMat(dataSet):
    """构建x+b系数矩阵：b默认为1"""
    m, n = shape(dataSet)
    dataMat = zeros((m, n))
    dataMat[:, 0] = 1                        # 矩阵第一列全为1-->b
    dataMat[:, 1:] = dataSet[:, :-1]         # 第二列到倒数第二列保持原数据，最后一列被删除
    return dataMat


def hardlim(dataSet):
    """激活函数：硬限幅函数"""
    dataSet[nonzero(dataSet.A > 0)[0]] = 1
    dataSet[nonzero(dataSet.A <= 0)[0]] = 0
    return dataSet


# 导入数据
Input = file2matrix("testSet.txt", "\t")
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
    gradient = dataMat * mat(weights)       # 梯度
    output = hardlim(gradient)              # 硬限幅函数预测分类
    errors = target - output                # 计算预测值与实际值之间的误差
    weights = weights + alpha*dataMat.T*errors
