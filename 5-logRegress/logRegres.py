from numpy import *
import random

def loadDataSet():
    dataSet = []
    label = []
    file = open("testSet.txt")
    for line in file.readlines():
        line = line.strip().split()
        dataSet.append([1.0, float(line[0]), float(line[1])])
        label.append(int(line[2]))
    return dataSet, label

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataSet, labelSet):
    data = mat(dataSet)
    label = mat(labelSet).transpose()
    m, n = shape(data)
    maxLoop = 500
    alpha = 0.001
    weights = ones((n, 1))
    for i in range(maxLoop):
        h = sigmoid(data * weights)
        weights = weights + alpha * data.transpose() * (label - h)
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+i+j)+0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            weights = weights + alpha * (classLabels[randIndex] - h) * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataSet, labelSet = loadDataSet()
    data = array(dataSet)
    m = shape(data)[0]
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(m):
        if labelSet[i] == 1:
            xcord1.append(data[i, 1])
            ycord1.append(data[i, 2])
        else:
            xcord2.append(data[i, 1])
            ycord2.append(data[i, 2])
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

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open("horseColicTraining.txt")
    frTest = open("horseColicTest.txt")
    trainSet = []
    trainLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainSet.append(lineArr)
        trainLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainSet), trainLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = float(errorCount)/numTestVec
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

def test():
    dataSet, labelSet = loadDataSet()
    weights = stocGradAscent1(array(dataSet), labelSet, 500)
    plotBestFit(weights)

multiTest()
