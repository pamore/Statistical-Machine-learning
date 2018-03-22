import csv
import numpy as np
import scipy.spatial.distance as distance
from collections import Counter
from operator import itemgetter
import matplotlib.pyplot as plt
import time
start_time = time.time()

train_row = csv.reader(open('D:/SML/assign2/Data/mnist_train.csv', "r"))
test_row = csv.reader(open('D:/SML/assign2/Data/mnist_test.csv', "r"))
k_List = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

dataRow1 = list(train_row)[:6000]
dataRow2 = list(test_row)[-1000:]

trainData = []
testData = []
for i in range(len(dataRow1)):
    trainData.append([float(x) for x in dataRow1[i]])

for j in range(len(dataRow2)):
    testData.append([float(x) for x in dataRow2[j]])

def euclidean_distance(pix_train, pix_test):
    return distance.euclidean(np.asarray(pix_train),np.asarray(pix_test),w=None)

total_point_dist = []
for i in range(len(testData)):
    one_Test_Point_dist = []
    one_test_px = testData[i][1:]
    for j in range(len(trainData)):
        one_Test_Point_dist.append([euclidean_distance(trainData[j][1:],one_test_px),trainData[j][0]])
    one_Test_Point_dist = sorted(one_Test_Point_dist, key=itemgetter(0))
    one_Test_Point_output = []
    for k in k_List:
        output = Counter([row[1] for row in one_Test_Point_dist[:k]]).most_common()
        correct = 1 if(output[0][0] == testData[i][0]) else 0
        # print "Correct->",correct
        one_Test_Point_output.append(correct)
    total_point_dist.append(one_Test_Point_output)
    # print "Done ", i

accuracy = []
testError = []
for p in range(len(k_List)):
    acc = float(sum([row[p] for row in total_point_dist]))/1000.0
    accuracy.append(acc)
    testError.append(1.0-acc)

plt.plot(k_List, testError, 'bo-')
plt.legend(('KNN Test Error'),loc='upper right')

plt.xlabel('# Nearest Neighbors')
plt.ylabel('Test Error')
plt.title('Error Rate of KNN Classifier')
print "--- %s seconds for completion ---" % (time.time() - start_time)
plt.show()