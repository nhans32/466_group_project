import sys
import pandas as pd
import numpy as np
from random import choice
from copy import copy, deepcopy
from dataStructs.C45Table import *
from classifier import *
from InduceC45 import c45
from helpers.randomForestHelpers import *
from helpers.randomForestHelpers import handleCommandLineParams
#import os

# Your Random Forest implementation shall be named randomForest.py or randomForest.java 
# (or a similar file name for a programming language of your choice). It shall take 
# as input the dataset filename, and the three input parameters described above. 
# It shall produce, as output, a results.txt or results.csv file that produces 
# predictions for each individual row in the dataset based on the 10-fold cross-validation 
# evaluation. Separately, it shall also output the confusion matrix and the accuracy 
# of prediction.


def selectRandom(table, numDataPoints, numAttributes):
    ''' Selects a random number of attributes from the table without replacment
    then selects a random number of rows from the table with replacment. Returns
    the modified table object
    '''
    # Select the proper number of data points
    table.trainingDf = table.trainingDf.sample(numDataPoints, replace=True)

    # This function call is slow, but it does keep recalculating the attribute info
    # Might need to refactor if too slow
    for _ in range(len(table.attributes) - numAttributes):
        rmAttr = choice(table.attributes)
        table.delAttribute(rmAttr)

    return table
    
    
def runC45onTable(table, threshold):
    ''' A wrapper function that collapese the table structure into something
    our C45 function can proccess. Takes in a table object, calls C45 function 
    one the decayed table, then returns the resulting tree.
    '''
    THRESHOLD = threshold
    # get probability of each unique value of target_var
    target_var_prob = table.trainingDf[table.classVar].value_counts(normalize=True)
    # calculate entropy of target_var
    target_var_entropy = -np.sum(target_var_prob * np.log2(target_var_prob))

    # special concideration to remove the domain size for the class attribute
    domain_sizes = copy(table.domain_sizes)
    del domain_sizes[table.classVar]
    domain_sizes = list(domain_sizes.values())

    # Finally ready to call c45
    tree = c45(table.trainingDf, table.attributes, domain_sizes, table.classVar, table.domain_sizes[table.classVar], target_var_entropy, THRESHOLD)
    return tree



if __name__ == "__main__":
    # If testing mode is enabled, only these hard-coded valus are used, no command line arguments
    TESTING_MODE = True
    VERBOSE = True
    dataFile = '../data/student-mat.csv'
    numAttributes = 10
    numDataPoints = 100
    numTrees = 50
    THRESHOLD = 0.01

    overallCorrect = 0
    overallClassed = 0
    overallAccuracy = 0
    genConfMatrix = pd.DataFrame([])

    if not TESTING_MODE:
        dataFile, numAttributes, numDataPoints, numTrees = handleCommandLineParams(sys.argv)

    # Generate the datatable structure
    table = C45Table()
    success = table.buildFromCSV(dataFile)
    if success == -1: exit("Table did not build properly. Fix csv error above and try agin.")

    # create file called out/randomForests/results.txt
    outFile = open("question4_out/randomForests/results.txt", "w")

    # Generate 10 folds of data
    srcDF = table.trainingDf
    dfsize = len(srcDF)
    srcDFRandom = srcDF.sample(frac=1)
    folds = np.array_split(srcDFRandom, 10)

    for idx, holdoutset in enumerate(folds):
        if len(folds) == 1:
            trainingSet = holdoutset
        else:
            testSetHoldRem = folds.copy()
            testSetHoldRem.pop(idx)
            trainingSet = pd.concat(testSetHoldRem)
        table.trainingDf = trainingSet
        # Generate the given number of trees
        forest = []
        if VERBOSE: print(f'C45 TODO: {"█" * numTrees}\nPROGRESS: ', end ='')
        for _ in range(numTrees):
            table_cpy = deepcopy(table)
            table_cpy = selectRandom(table_cpy, numDataPoints, numAttributes)
            # Call C45 on the randomly designed tree, hopfully the function will take this
            result = runC45onTable(table_cpy, THRESHOLD)
            forest.append((table_cpy,result))
            if VERBOSE: print('█', end ='', flush = True)

        # Now run the classifier on the random forest (always 10 fold)
        #
        # It shall produce, as output, results.txt with predictions for each individual row in 
        # the dataset based on the  10-fold cross-validation evaluation.
        #
        # Separately, it shall also output the confusion matrix and the accuracy of prediction (TO CONSOLE)
        testTable = deepcopy(table)
        testTable.trainingDf = holdoutset

        classed, correctClassed, incorrectClassed, accuracy, confusionMatrix = classifyTrainingSetForest(forest, testTable, True, False, outFile)
        
        genConfMatrix = pd.concat([genConfMatrix, confusionMatrix], axis=0)
        genConfMatrix = genConfMatrix.groupby(level=0).sum()
        
        overallClassed += classed
        overallCorrect += correctClassed
        overallAccuracy += accuracy

        print("Fold " + str(idx+1) + ':', end=' ')
        outFile.write("Fold " + str(idx+1) + ': ')
        print("Correct: " + str(correctClassed), end=' ')
        outFile.write("Correct: " + str(correctClassed) + " ")
        print("Incorrect: " + str(incorrectClassed))
        outFile.write("Incorrect: " + str(incorrectClassed) + "\n")
        print("===========================")

    outFile.write("===========================\n")
    print("Confusion Matrix: ")
    outFile.write("Confusion Matrix: \n")
    print(genConfMatrix)
    outFile.write(str(genConfMatrix) + "\n")
    print("===========================")
    outFile.write("===========================\n")
    print("Total Classed: ", overallClassed)
    print("Overall Accuracy: ", end='')
    print((overallCorrect / overallClassed)*100)
    outFile.write("Overall Accuracy: ")
    outFile.write(str((overallCorrect / overallClassed)*100) + "\n")
    print("Average Accuracy: ", end='')
    print(overallAccuracy/10)
    outFile.write("Average Accuracy: ")
    outFile.write(str(overallAccuracy/10) + "\n")

    outFile.close()