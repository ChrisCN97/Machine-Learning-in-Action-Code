from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    # compute distance
    # sort distance
    m = dataSet.shape[0]
    distance = tile(inX, (m, 1)) - dataSet
    distance = sum(distance**2, axis=1)**0.5
    sortDisIndex = distance.argsort()
    classCount = {}
    for i in range(k):
        classT = labels[sortDisIndex[i]]
        classCount[classT] = classCount.get(classT, 0) + 1
    classCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return classCount[0][0]

def file2matrix(filename):
    file = open(filename)
    lineList = file.readlines()
    m = len(lineList)
    dataSet = zeros((m, 3))
    labels = []
    index = 0
    for line in lineList:
        line = line.strip().split('\t')
        dataSet[index, :] = line[0:3]
        labels.append(int(line[3]))
        index += 1
    return dataSet, labels

def draw(dataSet, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[:, 0], dataSet[:, 1], 15.0*array(labels), 15.0*array(labels))
    plt.show()

def autoNorm(dataSet):
    minV = dataSet.min(0)
    maxV = dataSet.max(0)
    ranges = maxV - minV
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minV, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minV

def test():
    testRatio = 0.1
    dataSet, labels = file2matrix("datingTestSet2.txt")
    dataSet, ranges, minV = autoNorm(dataSet)
    m = dataSet.shape[0]
    numTest = int(m*testRatio)
    errCount = 0
    for i in range(numTest):
        res = classify0(dataSet[i, :], dataSet[numTest:m, :], labels[numTest:m], 3)
        print("classify: %d, real: %d" % (res, labels[i]))
        if res != labels[i]:
            errCount += 1
    print("Error rate: %f" % (errCount/float(numTest)))
    # draw(dataSet, labels)

def img2vector(filename):
    returnVect = zeros((1, 1024))
    file = open(filename)
    for i in range(32):
        line = file.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(line[j])
    return returnVect

def handwritingClassifyTest():
    hwLabels = []
    trainingFileList = listdir("digits/trainingDigits")
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        filename = trainingFileList[i]
        hwLabels.append(int(filename.split("_")[0]))
        trainingMat[i, :] = img2vector("digits/trainingDigits/%s" % filename)
    testFileList = listdir("digits/testDigits")
    tm = len(testFileList)
    errCount = 0
    for i in range(tm):
        filename = testFileList[i]
        label = int(filename.split("_")[0])
        testMat = img2vector("digits/testDigits/%s" % filename)
        res = classify0(testMat, trainingMat, hwLabels, 3)
        print("classify: %d, real: %d" % (res, label))
        if res != label:
            errCount += 1
    print("Error rate: %f" % (errCount / float(tm)))

# test()
handwritingClassifyTest()
