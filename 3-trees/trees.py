from math import log
import operator
import treePlotter

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    resDataSet = []
    for data in dataSet:
        if data[axis] == value:
            res = data[:axis]
            res.extend(data[axis+1:])
            resDataSet.append(res)
    return resDataSet

def calcShannonEnt(dataSet):
    m = len(dataSet)
    labelCount = {}
    for data in dataSet:
        label = data[-1]
        labelCount[label] = labelCount.get(label, 0) + 1
    entropy = 0.0
    for key in labelCount:
        pro = float(labelCount[key])/m
        entropy -= pro*log(pro, 2)
    return entropy

def chooseBestFeatureToSplit(dataSet):
    n = len(dataSet[0])-1
    m = len(dataSet)
    baseEnt = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(n):
        featureList = [data[i] for data in dataSet]
        featureSet = set(featureList)
        ent = 0.0
        for f in featureSet:
            resDataSet = splitDataSet(dataSet, i, f)
            resProb = float(len(resDataSet))/m
            ent += resProb*calcShannonEnt(resDataSet)
        infoGain = baseEnt - ent
        if infoGain > bestInfoGain:
            bestFeature = i
            bestInfoGain = infoGain
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for c in classList:
        classCount[c] = classCount.get(c, 0) + 1
    classCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return classCount[0][0]

def createTree(dataSet, labels):
    classList = [data[-1] for data in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeatureIndex = chooseBestFeatureToSplit(dataSet)
    bestFeature = labels[bestFeatureIndex]
    myTree = {bestFeature: {}}
    del(labels[bestFeatureIndex])
    featureList = [data[bestFeatureIndex] for data in dataSet]
    featureSet = set(featureList)
    for value in featureSet:
        subLabels = labels[:]
        resDataSet = splitDataSet(dataSet, bestFeatureIndex, value)
        myTree[bestFeature][value] = createTree(resDataSet, subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    res = ""
    for key in list(secondDict.keys()):
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                res = classify(secondDict[key], featLabels, testVec)
            else:
                res = secondDict[key]
            break
    return res

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


def test():
    dataSet, labels = createDataSet()
    # sdataSet = splitDataSet(dataSet, 0, 1)
    # ent = calcShannonEnt(dataSet)
    # bestFeature = chooseBestFeatureToSplit(dataSet)
    # myTree = createTree(dataSet, labels)
    myTree = treePlotter.retrieveTree(0)
    print(classify(myTree, labels, [1, 1]))

test()
