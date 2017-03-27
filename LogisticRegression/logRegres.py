# coding:utf8
# Logistic Regression Working Module
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    """打开文本文件并逐行读取"""
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()  # 默认以tab分割
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # X0，X1，X2
        labelMat.append(int(lineArr[2]))  # 类别标签
    return dataMat, labelMat


def sigmoid(inX):
    # sigmoid阶跃函数
    return 1.0/(1+exp(-inX))


def gradAscent(dataMatIn, classLabels):
    """梯度上升算法"""
    dataMatrix = mat(dataMatIn)              # convert to NumPy matrix
    labelMat = mat(classLabels).transpose()  # 转换为列向量
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):               # heavy on matrix operations
        h = sigmoid(dataMatrix*weights)      # matrix mult
        error = (labelMat - h)               # vector subtraction
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
    return weights


def plotBestFit(weights):
    """画出决策边界"""
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    """随机梯度上升算法"""
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   # initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """改进的随机梯度上升算法"""
    m, n = shape(dataMatrix)
    weights = ones(n)   # initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    # alpha decreases with iteration, does not go to 0 because of the constant
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))


if __name__ == "__main__":
    pass
