from numpy import *

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    res = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        res[dataMatrix[:, dimen] <= threshVal] = -1
    else:
        res[dataMatrix[:, dimen] > threshVal] = -1
    return res

def buildStump(dataArr, classLabels, D):
    data = mat(dataArr)
    classl = mat(classLabels).T
    m, n = shape(data)
    numStep = 10.0
    bestErr = inf
    bestRes = mat(zeros((m, 1)))
    bestStump = {}
    for i in range(n):
        rangeMin = data[:, i].min()
        rangeMax = data[:, i].max()
        rangeStep = (rangeMax-rangeMin)/numStep
        for step in range(-1, int(numStep)+1):
            for ineq in ['lt', 'gt']:
                threshVal = rangeMin + step*rangeStep
                res = stumpClassify(data, i, threshVal, ineq)
                errVec = mat(ones((m, 1)))
                errVec[res == classl] = 0
                wErr = D.T*errVec
                # print("split: dim %d, thresh %.2f, ineqal: %s, wErr: %.3f" % (i, threshVal, ineq, wErr))
                if wErr < bestErr:
                    bestErr = wErr
                    bestRes = res.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = ineq
    return bestStump, bestErr, bestRes