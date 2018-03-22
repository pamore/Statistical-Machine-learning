
import csv
import math
import random
import numpy
import matplotlib.pyplot as plt
from operator import mul

#Variables for plotting Accuracy
accuracyList = []
trainingSetSize = []

accuracyList2 = []
trainingSetSize2 = []

#Variables for generating 400 samples
#mean of each x_i at y=1 for all data
totalMeanPositive = [0 for i in range(4)]
#variance of each x_i at y=1 for all data
totalVariancePositive = [0 for i in range(4)]

def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1 / (1 + math.exp(gamma))
    else:
        return 1 / (1 + math.exp(-gamma))

def loadFile(file):

	row = csv.reader(open(file, "r"))
	dataRow = list(row)

	for i in range(len(dataRow)):
            dataRow[i] = [float(x) for x in dataRow[i]]
	return dataRow

fileName = 'D:/SML/assign1/data_banknote_authentication.csv'
fullData = loadFile(fileName)
#print "rows=",len(fullData)
#print "sample 1=",fullData[0]


def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

splitRatioList = [0.01, 0.02, 0.05, 0.1, 0.625, 1]
for splitRatio in splitRatioList:

    accuracyAvg = 0
    accuracyAvg2 = 0

    for accuracyAvgCount in range(5): # each training set tested 5 times

        trainData, testData = splitDataset(fullData, splitRatio)
        #print('Split {0} rows into train with {1} and test with {2}').format(len(fullData), len(trainData), len(testData))

        #training data -> trainData
        #testing data -> fullData

        #############   TRAINING DATA ####################

        def separateByClass(dataset):
            separated = {}
            for i in range(len(dataset)):
                vector = dataset[i]
                if (vector[-1] not in separated):
                    separated[vector[-1]] = []
                separated[vector[-1]].append(vector)
            return separated

        separated = separateByClass(trainData)

        ###############################################################
        #####################   GAUSSIAN NAIVE BAYES     ##############
        ###############################################################


        probabilityOutput = [] # [P(Y=1)  P(Y=0)]
        probabilityOutput.append(float(len(separated[1]))/len(trainData))
        probabilityOutput.append(float(len(separated[0]))/len(trainData))

        #print "P(1) and P(0) for training data is",probabilityOutput
        numAttributes = len(trainData[0])-1
        meanPositive = [0 for i in range(numAttributes)]
        meanNegative = [0 for i in range(numAttributes)]


        for i in range(numAttributes):
            s1=0
            for r1 in range(len(separated[1])):
               s1 += separated[1][r1][i]
            s1 = s1/len(separated[1])
            meanPositive[i] = s1
            s2 = 0
            for r2 in range(len(separated[0])):
               s2 += separated[0][r2][i]
            s2 = s2/len(separated[0])
            meanNegative[i] = s2

        #print 'meanPositive=',meanPositive
        #print 'meanNeg=',meanNegative
        mean = [meanPositive , meanNegative]

        stdDevPositive = [0 for i in range(numAttributes)]
        stdDevNegative = [0 for i in range(numAttributes)]

        for i in range(numAttributes):
            s1=0
            for r1 in range(len(separated[1])):
               s1 += math.pow((separated[1][r1][i]-meanPositive[i]),2)
            s1 = s1/len(separated[1])
            stdDevPositive[i] = s1

            s2 = 0
            for r2 in range(len(separated[0])):
               s2 += math.pow((separated[0][r2][i]-meanNegative[i]),2)
            s2 = s2/len(separated[0])
            stdDevNegative[i] = s2

        #print 'stdDevPos=',stdDevPositive
        #print 'stdDevNeg=',stdDevNegative
        stdDev = [stdDevPositive ,stdDevNegative]
        #print 'std=',stdDev

        def gnbPerAttribute(std,mean,xi):
            prob = 0
            prob = math.exp((math.pow((xi-mean),2))/(-2*std))/math.pow((std*2*math.pi),0.5)
            return prob

        ############### TESTING ##########################

        probPositiveAttribute = [0,0,0,0]
        probNegativeAttribute = [0,0,0,0]
        OutputActualExpected = [] #### output = [[Actual Expected], [Actual Expected].....]
        for row in fullData:
            for i in range(numAttributes):
                probPositiveAttribute[i] = gnbPerAttribute(stdDevPositive[i], meanPositive[i],row[i])
                probNegativeAttribute[i] = gnbPerAttribute(stdDevNegative[i], meanNegative[i],row[i])
            #print 'probPositiveAttribute=',probPositiveAttribute
            ProbCombinedPositive = numpy.prod(probPositiveAttribute)    ###P(x1,x1,x3,x4/y=1)
            #print 'ProbCombinedPositive=',ProbCombinedPositive
            ProbCombinedNegative = numpy.prod(probNegativeAttribute)    ###P(x1,x1,x3,x4/y=0)

            ########             BAYES   RULE       #############

            ProbTrue = (ProbCombinedPositive*probabilityOutput[0])/((ProbCombinedPositive*probabilityOutput[0])+(ProbCombinedNegative*probabilityOutput[1]))
            ProbFalse = (ProbCombinedNegative*probabilityOutput[1])/((ProbCombinedPositive*probabilityOutput[0])+(ProbCombinedNegative*probabilityOutput[1]))
            if ProbTrue>ProbFalse:
                OutputActualExpected.append([1.0, row[-1] ])
            else:
                OutputActualExpected.append([0.0, row[-1]])

        #print 'Output',OutputActualExpected
        accuracy = 0
        correctPred = 0
        for row in OutputActualExpected:
            if row[0]==row[1]:
                correctPred += 1
        accuracy = float(correctPred)/len(OutputActualExpected)
        #print ('accuracy for count{0}={1}').format(accuracyAvgCount,accuracy)
        accuracyAvg += accuracy



        if splitRatio == 1 and accuracyAvgCount==4:
            totalMeanPositive = meanPositive[:]
            totalVariancePositive = [math.sqrt(p) for p in stdDevPositive]


        #####################                           #####################
        #####################   LOGISTIC RESGRESSION    #####################
        #####################                           #####################

        weights = [1 for i in range(numAttributes+1)] # w0,w1,w2,w3,w4
        old_Weights = []

        learningRate = 10

        probSigmoid = []
        probSigmoidTest = []

        threshold=5
        iteration = 0

        while iteration < 300:
            iteration+=1
            old_Weights = list(weights)
            # print '#Iteration=',iteration
            for row in trainData:
                temp = sum(map(mul,weights[1:],row[0:-1]))+weights[0]
                #print 'temp=', temp
                probSigmoid.append(sigmoid(temp))

            #######W0 = calculate
            errorSum = 0
            for r in range(len(trainData)):
                errorSum += (trainData[r][-1] - probSigmoid[r])
            # print 'errorSum=',errorSum
            wTemp = weights[0] + (learningRate * (errorSum))
        #     threshold = min(diffSumMax, abs(diffSum))
            weights[0] = wTemp

            #########  W1,W2,W3,W4 calculate
            for k in range(1,len(weights)):
                wTemp = 0
                errorSum = 0
                for r in range(len(trainData)):
                    errorSum += (trainData[r][-1]-probSigmoid[r])*trainData[r][k-1]
                # print 'errorSum=', errorSum
                wTemp = weights[k] + (learningRate*(errorSum))
        #         threshold = min(threshold, abs(diffSum))
                weights[k] = wTemp
            if cmp(old_Weights, weights) == 0:
                break

        accuracy2 = 0

        ####### TESTING
        for row in fullData:
            temp = sum(map(mul, weights[1:], row[0:-1])) + weights[0]
            # print 'temp=', temp
            result = sigmoid(temp)
            if (row[-1]-result<0.5):
                accuracy2+=1
        accuracy2 = float(accuracy2)/len(fullData)
        #print 'accuracy LR=',accuracy2
        accuracyAvg2 += accuracy2

        ##################################################
        ###  END
        ##################################################

    accuracyList.append(accuracyAvg*100/5)
    trainingSetSize.append(len(trainData))

    accuracyList2.append(accuracyAvg2 * 100 / 5)
    trainingSetSize2.append(len(trainData))


########################################################
#############   GENERATIVE MODEL  ######################
########################################################
sampleList = []
for i in range(numAttributes):
    sampleList.append(list(numpy.random.normal(totalMeanPositive[i], totalVariancePositive[i], 400)))

print 'meanPositiveOLD=',totalMeanPositive
print 'varPositiveOLD=',totalVariancePositive

totalMeanPositiveNew = [0 for i in range(numAttributes)]
for i in range(numAttributes):
    s1=0
    for x1 in sampleList[i]:
       s1 += x1
    s1 = s1/len(sampleList[i])
    totalMeanPositiveNew[i] = s1

print 'meanPositiveNew=',totalMeanPositiveNew

totalVariancePositiveNew = [0 for i in range(numAttributes)]
for i in range(numAttributes):
    s1=0
    for x1 in range(len(sampleList[i])):
       s1 += math.pow((sampleList[i][x1]-totalMeanPositiveNew[i]),2)

    s1 = s1/len(sampleList[i])
    totalVariancePositiveNew[i] = s1

print 'varPositiveNew=',totalVariancePositiveNew

fig, ax = plt.subplots()
ax.scatter(totalMeanPositiveNew, totalVariancePositiveNew)
ax.scatter(totalMeanPositive, totalVariancePositive)

enum = ['x1','x2','x3','x4']
for i, txt in enumerate(enum):
    ax.annotate(txt, (totalMeanPositiveNew[i],totalVariancePositiveNew[i]))
    ax.annotate(txt, (totalMeanPositive[i],totalVariancePositive[i]))
plt.legend(('NEW SAMPLE DATA', 'ORIGINAL DATA'),loc='upper right')
plt.xlabel('mean of each attribute at y=1')
plt.ylabel('varience of each attribute at y=1')
plt.title('Mean v/s Varience for y=1 (for each attribute)')

##################################################
#############    PLOT ACCURACY   #################
##################################################
# print 'accuracyList=',accuracyList
# print 'trainingSetSize=',trainingSetSize
#
# print 'accuracyList2=',accuracyList2
# print 'trainingSetSize2=',trainingSetSize2
plt.figure(2)
plt.plot(trainingSetSize, accuracyList, 'bo-')
plt.plot(trainingSetSize2, accuracyList2, 'ro-')
plt.legend(('Gaussian NB', 'Logistic Regression'),loc='upper right')

plt.xlabel('Training Set Size(row count)')
plt.ylabel('Accuracy(%)')
plt.title('Learning curve for GNB and LR')
plt.show()

