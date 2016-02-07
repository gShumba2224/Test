import numpy as np
import operator
import Item

trainingSetFIle = ("/home/gman/PycharmProjects/untitled/Machine Learning in Action Resources/Ch02/datingTestSet.txt")
trainingMat, labels = Item.getData(trainingSetFIle)

def kNN (inputVector,trainingMat, labels, k):
    diffMat = np.tile(inputVector,(trainingMat.shape[0],1))-trainingMat
    sqDiffMat = np.asarray(diffMat)**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    labelDict = {}
    count = 0
    for distance in distances:
        labelDict[distance]= labels[count]
        count = count + 1
    distances = np.sort(distances)
    classProbability={}
    for i in range (k):
        currentLabel = labelDict [distances[i]]
        classProbability[currentLabel]=classProbability.get(currentLabel,0)+1

    highest = classProbability.keys()[0]
    for i in classProbability.keys():
        if (classProbability[highest]<classProbability[i]):
            highest = i
    print highest



x = np.array([43673,1.889,0.191])
y = np.matrix([[6,5,4],[3,2,1]])
kNN(x,trainingMat,labels,10)

