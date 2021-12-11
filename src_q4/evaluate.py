import sys
from helpers.evaluateHelpers import *
from InduceC45 import c45, read_csv
from classifier import classifyTrainingSet
import pandas as pd
import numpy as np
import os
import time

'''
You shall perform cross-validation analysis of the accuracy of your classifiers 
using the training set data made available to you.

To that extent, you shall implement a program validation.java/evaluate.py, 
which will take as input the training file, the optional restrictions file and 
an integer number n specifying how many-fold the cross-validation has to be.

n = 0 represents no cross-validation (i.e., use entire training set to construct 
a single classifier), while n = −1 represents all-but-one cross-validation.

The Validation.java program shall perform the n-fold cross-validation evaluation 
of your InduceC45 imple- mentation. It shall produce as output the following 
information:
1. The overall confusion matrix.
2. The overall recall, precision, pf and f-measure (compute these numbers with 
   respect to recognizing Obama’s voters).
3. Overall and average accuracy (two-way) and error rate of prediction.
'''


def runEvaluate(df, attributes, attribute_domains, target_var, target_domain, n, threshold, filepath):
    # output file creation
    fileNameArr = filepath.split('/')
    outFileName = "out/evaluateOutput/" + fileNameArr[len(fileNameArr) - 1].split('.csv')[0] + "-results.out"

    if os.path.exists(outFileName):
        os.remove(outFileName)
    file = open(outFileName, "w+")

    srcDF = df
    dfsize = len(srcDF)

    srcDFRandom = srcDF.sample(frac=1)
    if n == 0:
        testSets = np.array_split(srcDFRandom, 1)
    elif n == -1:
        testSets = np.array_split(srcDFRandom, dfsize)
    else:
        testSets = np.array_split(srcDFRandom, n)

    overallCorrect = 0
    overallClassed = 0
    overallAccuracy = 0
    genConfMatrix = pd.DataFrame([])

    numFolds = len(testSets)
    # k-fold validation
    for idx, holdoutset in enumerate(testSets):
        # creating training set for all sets that are not the holdout set
        # if we are not doing cross validation
        if len(testSets) == 1:
            trainingSet = holdoutset
        else:
            testSetHoldRem = testSets.copy()
            testSetHoldRem.pop(idx)
            trainingSet = pd.concat(testSetHoldRem)

        # CREATE CLASSIFIER
        # get entropy of target variable
        target_var_prob = trainingSet[target_var].value_counts(normalize=True)
        target_var_entropy = -np.sum(target_var_prob * np.log2(target_var_prob))
        # create model for classification
        # time start 
        start = time.time()
        modelTree = c45(trainingSet, attributes, attribute_domains, target_var, target_domain, target_var_entropy, threshold)
        # time end
        end = time.time()
        print("Time taken to build tree: " + str(end - start))

        # time
        startTime = time.time()
        classed, correctClassed, incorrectClassed, accuracy, confusionMatrix = classifyTrainingSet(modelTree, holdoutset, 
                                                                                    attributes, attribute_domains, target_var, target_domain, True, True)
        endTime = time.time()
        print("Time taken for classification: " + str(endTime - startTime))

        genConfMatrix = pd.concat([genConfMatrix, confusionMatrix], axis=0)
        genConfMatrix = genConfMatrix.groupby(level=0).sum()
        
        overallClassed += classed
        overallCorrect += correctClassed
        overallAccuracy += accuracy

        print("Fold " + str(idx+1) + ':', end=' ')
        file.write("Fold " + str(idx+1) + ': ')
        print("Correct: " + str(correctClassed), end=' ')
        file.write("Correct: " + str(correctClassed) + " ")
        print("Incorrect: " + str(incorrectClassed))
        file.write("Incorrect: " + str(incorrectClassed) + "\n")

    print("===========================")
    file.write("===========================\n")
    print("Confusion Matrix: ")
    file.write("Confusion Matrix: \n")
    print(genConfMatrix)
    file.write(str(genConfMatrix) + "\n")
    print("===========================")
    file.write("===========================\n")
    print("Total Classed: ", overallClassed)
    print("Overall Accuracy: ", end='')
    print((overallCorrect / overallClassed)*100)
    file.write("Overall Accuracy: ")
    file.write(str((overallCorrect / overallClassed)*100) + "\n")
    print("Average Accuracy: ", end='')
    print(overallAccuracy/numFolds)
    file.write("Average Accuracy: ")
    file.write(str(overallAccuracy/numFolds) + "\n")

    file.close()


if __name__ == "__main__":
    TESTING_MODE = True
    TRAINING_CSV = "/Users/otakar/cs/466/466_group_project/data/student-mat.csv"
    RESTRICTIONS_CSV = None
    THRESHOLD = 0.01
    n = 10

    if TESTING_MODE:
        trainingPath, restrPath, n = handleCommandLineParams(["test", TRAINING_CSV, RESTRICTIONS_CSV, n])
    else:
        trainingPath, restrPath, n = handleCommandLineParams(sys.argv)


    df, attributes, attribute_domains, target_var, target_domain = read_csv(trainingPath, restrPath)
    runEvaluate(df, attributes, attribute_domains, target_var, target_domain, n, THRESHOLD, trainingPath)
