from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set()  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    m = len(trainMatrix)
    n = len(trainMatrix[0])
    pAubsive = sum(trainCategory)/float(m)
    p0Num = ones(n)
    p1Num = ones(n)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(m):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vec = log(p1Num/p1Denom)
    p0Vec = log(p0Num / p0Denom)
    return p0Vec, p1Vec, pAubsive

def classifyNB(vec, p0Vec, p1Vec, pAubsive):
    p0 = sum(vec*p0Vec)+log(1-pAubsive)
    p1 = sum(vec*p1Vec)+log(pAubsive)
    if p1 > p0:
        return 1
    else:
        return 0

def textParse(s):
    import re
    listOfTokens = re.split(r'\W+', s)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def testSpam():
    docList = []
    classList = []
    for i in range(1, 26):
        wordList = textParse(open("email/spam/%d.txt" % i).read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open("email/ham/%d.txt" % i, encoding='ISO-8859-15').read())  # gbk err
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))  # python3
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainingMat = []
    trainingClass = []
    for docIndex in trainingSet:
        trainingMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainingClass.append(classList[docIndex])
    p0Vec, p1Vec, pSpam = trainNB0(array(trainingMat), array(trainingClass))
    err = 0
    for testIndex in testSet:
        res = classifyNB(setOfWords2Vec(vocabList, docList[testIndex]), p0Vec, p1Vec, pSpam)
        if res != classList[testIndex]:
            err += 1
    print("err rate:", float(err)/len(testSet))

def test():
    postingList, classVec = loadDataSet()
    vocabList = createVocabList(postingList)
    trainMatrix = []
    for data in postingList:
        trainMatrix.append(setOfWords2Vec(vocabList, data))
    p0Vec, p1Vec, pAubsive = trainNB0(trainMatrix, classVec)
    test = ["love", "my", "dalmation"]
    testVec = array(setOfWords2Vec(vocabList, test))
    print(test, "classify as:", classifyNB(testVec, p0Vec, p1Vec, pAubsive))
    test = ["stupid", "garbage"]
    testVec = array(setOfWords2Vec(vocabList, test))
    print(test, "classify as:", classifyNB(testVec, p0Vec, p1Vec, pAubsive))

# test()
testSpam()