def ToMMA(transMat):
    h = len(transMat)
    w = len(transMat[0])
    for i in range(h):
        oline = "{"
        for j in range(w):
            oline += transMat[i][j]
            if j != w - 1:
                oline += ", "
            else:
                oline += "},"
        print(oline)


nStateOfLoss = 2
nStateOfError = 5
statesLoss = ["l" + str(i) for i in range(nStateOfLoss)]
statesError = ["e" + str(i) for i in range(nStateOfError)]
transitMat = [["0" for _ in range(nStateOfError * nStateOfLoss)] for _ in range(nStateOfError * nStateOfLoss)]
for i in range(nStateOfLoss):
    line = []
    for j in range(nStateOfError):
        currentState = i * nStateOfError + j
        # go to itself
        transitMat[currentState][currentState] = "p" + statesLoss[i] + statesError[j] + "_" + statesLoss[i] + \
                                                 statesError[j]
        # i and j within bound
        if 0 < i < nStateOfLoss - 1 and 0 < j < nStateOfError - 1:
            transitMat[currentState][currentState - 1] = "p" + statesLoss[i] + statesError[j] + "_" + statesLoss[i] + \
                                                         statesError[j - 1]
            transitMat[currentState][currentState + 1] = "p" + statesLoss[i] + statesError[j] + "_" + statesLoss[i] + \
                                                         statesError[j + 1]
            transitMat[currentState][currentState + nStateOfError] = "p" + statesLoss[i] + statesError[j] + "_" + \
                                                                     statesLoss[i + 1] + statesError[j]
            transitMat[currentState][currentState - nStateOfError] = "p" + statesLoss[i] + statesError[j] + "_" + \
                                                                     statesLoss[i - 1] + statesError[j]
        # i at left bound , j within bound
        elif i == 0 and 0 < j < nStateOfError - 1:
            transitMat[currentState][currentState - 1] = "p" + statesLoss[i] + statesError[j] + "_" + statesLoss[i] + \
                                                         statesError[j - 1]
            transitMat[currentState][currentState + 1] = "p" + statesLoss[i] + statesError[j] + "_" + statesLoss[i] + \
                                                         statesError[j + 1]
            transitMat[currentState][currentState + nStateOfError] = "p" + statesLoss[i] + statesError[j] + "_" + \
                                                                     statesLoss[i + 1] + statesError[j]
        # i at right bound , j within bound
        elif i == nStateOfLoss - 1 and 0 < j < nStateOfError - 1:
            transitMat[currentState][currentState - 1] = "p" + statesLoss[i] + statesError[j] + "_" + statesLoss[i] + \
                                                         statesError[j - 1]
            transitMat[currentState][currentState + 1] = "p" + statesLoss[i] + statesError[j] + "_" + statesLoss[i] + \
                                                         statesError[j + 1]
            transitMat[currentState][currentState - nStateOfError] = "p" + statesLoss[i] + statesError[j] + "_" + \
                                                                     statesLoss[i - 1] + statesError[j]
        # i within bound , j at left bound
        elif 0 < i < nStateOfLoss - 1 and j == 0:
            transitMat[currentState][currentState + 1] = "p" + statesLoss[i] + statesError[j] + "_" + statesLoss[i] + \
                                                         statesError[j + 1]
            transitMat[currentState][currentState + nStateOfError] = "p" + statesLoss[i] + statesError[j] + "_" + \
                                                                     statesLoss[i + 1] + statesError[j]
            transitMat[currentState][currentState - nStateOfError] = "p" + statesLoss[i] + statesError[j] + "_" + \
                                                                     statesLoss[i - 1] + statesError[j]
        # i at left bound , j at left bound
        elif i == 0 and j == 0:
            transitMat[currentState][currentState + 1] = "p" + statesLoss[i] + statesError[j] + "_" + statesLoss[i] + \
                                                         statesError[j + 1]
            transitMat[currentState][currentState + nStateOfError] = "p" + statesLoss[i] + statesError[j] + "_" + \
                                                                     statesLoss[i + 1] + statesError[j]
        # i at right bound , j at left bound
        elif i == nStateOfLoss - 1 and j == 0:
            transitMat[currentState][currentState + 1] = "p" + statesLoss[i] + statesError[j] + "_" + statesLoss[i] + \
                                                         statesError[j + 1]
            transitMat[currentState][currentState - nStateOfError] = "p" + statesLoss[i] + statesError[j] + "_" + \
                                                                     statesLoss[i - 1] + statesError[j]
        # i within bound , j at right bound
        elif 0 < i < nStateOfLoss - 1 and j == nStateOfError - 1:
            transitMat[currentState][currentState - 1] = "p" + statesLoss[i] + statesError[j] + "_" + statesLoss[i] + \
                                                         statesError[j - 1]
            transitMat[currentState][currentState - nStateOfError] = "p" + statesLoss[i] + statesError[j] + "_" + \
                                                                     statesLoss[i - 1] + statesError[j]
            transitMat[currentState][currentState + nStateOfError] = "p" + statesLoss[i] + statesError[j] + "_" + \
                                                                     statesLoss[i + 1] + statesError[j]
        # i at left bound , j at right bound
        elif i == 0 and j == nStateOfError - 1:
            transitMat[currentState][currentState - 1] = "p" + statesLoss[i] + statesError[j] + "_" + statesLoss[i] + \
                                                         statesError[j - 1]
            transitMat[currentState][currentState + nStateOfError] = "p" + statesLoss[i] + statesError[j] + "_" + \
                                                                     statesLoss[i + 1] + statesError[j]
        # i at right bound , j at right bound
        elif i == nStateOfLoss - 1 and j == nStateOfError - 1:
            transitMat[currentState][currentState - 1] = "p" + statesLoss[i] + statesError[j] + "_" + statesLoss[i] + \
                                                         statesError[j - 1]
            transitMat[currentState][currentState - nStateOfError] = "p" + statesLoss[i] + statesError[j] + "_" + \
                                                                     statesLoss[i - 1] + statesError[j]
ToMMA(transitMat)
