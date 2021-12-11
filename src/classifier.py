import sys
from helpers.classifierHelpers import *
import numpy as np

'''
Write a program classify.java/classify.py that will take as input the JSON 
description of a decision tree generated by your InduceC45.java/InduceC45.py 
program and a CSV file of records to be classified and outputs the 
classification result for each vector.
Classify.java is run as follows:
java Classifier <CSVFile> <JSONFile>
'''

# attrList may not be needed in this function


def findClass(tree, row, attributes, attribute_domains, isTuple = False):
    if tree.leaf == True: return tree.var
    else:
        if isTuple: rowAttrVal = getattr(row, tree.var)
        else: rowAttrVal = row[tree.var]
        for edge in tree.children:
            if edge.continuous == True:
                if rowAttrVal <= edge.value and edge.direction == 'le':
                    return findClass(edge.endNode, row, attributes, attribute_domains, isTuple)
                elif rowAttrVal > edge.value and edge.direction == 'gt':
                    return findClass(edge.endNode, row, attributes, attribute_domains, isTuple)
            elif rowAttrVal == edge.value:
                return findClass(edge.endNode, row, attributes, attribute_domains, isTuple)
        # find edge in tree.children with value None, will be ghost edge -> this may not be efficient
        for edge in tree.children:
            if edge.value == None:
                return findClass(edge.endNode, row, attributes, attribute_domains, isTuple)

def classifyReg(tree, dataframe, attributes, attribute_domains):
    print("Classification on non-Training Set:")
    for index, row in dataframe.iterrows():
        prediction = findClass(tree, row, attributes, attribute_domains)
        print("-----------------------------------")
        print("Entry: ", end='')
        print(row)
        print("Classification Prediction: ", end='')
        print(prediction)

def getClassValsFromTreeSet(tree, uniqueClassValsTree):
    if tree.leaf == True:
        uniqueClassValsTree.add(tree.var)
    else:
        for edge in tree.children:
            getClassValsFromTreeSet(edge.endNode, uniqueClassValsTree)
    return uniqueClassValsTree

def classifyTrainingSet(tree, dataframe, attributes, attribute_domains, target_var, target_domain, silent, evalSilent):
    classed = 0
    incorrectClassed = 0
    correctClassed = 0

    # construct empty confusion matrix
    confusionMatrix = pd.DataFrame([])
    # add y axis label to confusion matrix
    confusionMatrix.index.name = 'Actual'
    # add x axis label to confusion matrix
    confusionMatrix.columns.name = 'Predicted'

    if evalSilent == False:
        print("Classification on Training Set:")
    # go through df and classify
    for index, row in dataframe.iterrows():
        prediction = findClass(tree, row, attributes, attribute_domains, False)
        if silent == False:
            print("-----------------------------------")
            print("Entry: ", end='')
            print(row)
            print("Classification Prediction: ", end='')
            print(prediction)

        # populate confusion matrix
        if prediction not in confusionMatrix.columns:
            confusionMatrix[prediction] = 0
        if row[target_var] not in confusionMatrix.index:
            confusionMatrix.loc[row[target_var]] = 0

        confusionMatrix[prediction][row[target_var]] += 1

        # calculating stats
        if prediction == row[target_var]:
            correctClassed += 1
        else:
            incorrectClassed += 1
        classed += 1

    accuracy = correctClassed/classed
    errorRate = incorrectClassed/classed

    # print stats
    if evalSilent == False:
        print("========================================")
        print("Entries Classed (TOTAL): " + str(classed))
        print("Entries Correctly Classed: " + str(correctClassed))
        print("Entries Incorrectly Classed: " + str(incorrectClassed))
        print("Classification Accuracy: " + str(accuracy*100) + "%")
        print("Classification Error Rate: " + str(errorRate*100) + "%")

    if evalSilent == False:
        # print confusion matrix
        print("========================================")
        print("Confusion Matrix:")
        print(confusionMatrix)
        print("----------")

    return classed, correctClassed, incorrectClassed, accuracy*100, confusionMatrix


def classifyTrainingSetForest(forestAndTables, evalTable, silent, evalSilent, outfile):
    classed = 0
    incorrectClassed = 0
    correctClassed = 0

    # construct empty confusion matrix
    confusionMatrix = pd.DataFrame([])
    # add y axis label to confusion matrix
    confusionMatrix.index.name = 'Actual'
    # add x axis label to confusion matrix
    confusionMatrix.columns.name = 'Predicted'

    # get unique values of class variable to create confusion matrix
    # gathering unique classvar vals from the tree as well as from values in df to be classed
    uniqueClassVar = set(evalTable.domain_opts[evalTable.classVar])
    for _, treeIter in forestAndTables:
        uniqueClassVar = getClassValsFromTreeSet(treeIter, uniqueClassVar)

    if not evalSilent: print("\nClassification on Training Set:")
    
    # go through df and classify
    for row in evalTable.trainingDf.itertuples():
        predictions = []
        for tableIter, treeIter in forestAndTables:
            prediction = findClass(treeIter, row, tableIter.attributes, tableIter.domain_opts, True)
            predictions.append(prediction)
        prediction = max(set(predictions), key=predictions.count)
        # write row and its prediction to outfile
        if outfile != None:
            # if prediction is np.int64, convert to int
            if isinstance(prediction, np.int64):    
                outfile.write(str(row) + "," + str(int(prediction)) + "\n")
            else:
                outfile.write(str(row) + "," + str(prediction) + "\n")

        if prediction is None: exit(f'FAILED: {row}')
        if not silent:
            print(f'-----------------------------------\nEntry: {row}\nClassification Prediction: {prediction}')

        # populate confusion matrix
        if prediction not in confusionMatrix.columns:
            confusionMatrix[prediction] = 0
        if getattr(row, evalTable.classVar) not in confusionMatrix.index:
            confusionMatrix.loc[getattr(row, evalTable.classVar)] = 0

        confusionMatrix[prediction][getattr(row, evalTable.classVar)] += 1

        # calculating stats
        if prediction == getattr(row, evalTable.classVar):
            correctClassed += 1
        else:
            incorrectClassed += 1
        classed += 1

    accuracy = correctClassed/classed
    errorRate = incorrectClassed/classed

    # print stats
    if evalSilent == False:
        # print stats
        print("========================================")
        print("Entries Classed (TOTAL): " + str(classed))
        print("Entries Correctly Classed: " + str(correctClassed))
        print("Entries Incorrectly Classed: " + str(incorrectClassed))
        print("Classification Accuracy: " + str(accuracy*100) + "%")
        print("Classification Error Rate: " + str(errorRate*100) + "%")
        # print confusion matrix
        print("========================================")
        print("Confusion Matrix:")
        print(confusionMatrix)
        print("----------")

    return classed, correctClassed, incorrectClassed, accuracy*100, confusionMatrix

if __name__ == "__main__":
    TESTING_MODE = True
    csvFileTEST = 'in\c45Input2\heart.csv'
    JSONFileTEST = 'question4_out\c45Output\c45Trees\heart_0.1.json'
    SILENT = True

    if TESTING_MODE:
        classify, tree = handleCommandLineParams(["test", csvFileTEST, JSONFileTEST])
    else:
        classify, tree = handleCommandLineParams(sys.argv)

    df, attributes, attribute_domains, target_var = classify[0]
    trainingSetBool = classify[1]

    if trainingSetBool == True:
        # ignore class variable in dataset
        # remove target var from attribute domains
        target_domain = attribute_domains.pop(attributes.index(target_var))
        # remove target var from attributes
        attributes.remove(target_var)
        classifyTrainingSet(tree, df, attributes, attribute_domains, target_var, target_domain, SILENT, False)
    else:
        classifyReg(tree, df, attributes, attribute_domains)
