import pandas as pd
import linecache
import json

# Prints the command structure from the instructions and then exits, nice.
def printHelpMenuExit():
    print('USAGE: python3 classifier.py [csvFile] [JSONFile]')
    exit()

# need more validation for file types
def handleCommandLineParams(arguments):
    if not (len(arguments) == 3 or len(arguments) == 4):
        print(f'Invalid number of arguments {len(arguments) - 1}, expected 3 or 4')
        printHelpMenuExit()
    else:
        trainingPath = arguments[1]
        restrictionsPath = None
        if len(arguments) == 3:
            n = int(arguments[2])
        elif len(arguments) == 4:
            restrictionsPath = arguments[2]
            n = int(arguments[3])
        return trainingPath, restrictionsPath, n


def constructDf(trainingPath, restrictionsPath):
    try:
        trainingDf = pd.read_csv(trainingPath, skiprows=[1, 2])

        header = linecache.getline(trainingPath, 1).strip().split(",")
        domains = list(map(int, linecache.getline(trainingPath, 2).strip().split(',')))
        classVar = linecache.getline(trainingPath, 3).strip()
        attrDict = {}
    except Exception as e:
        exit(f'ERR: trainingSetFile: {e}')

    restrVector = []
    if restrictionsPath != None:
        try:
            restrFile = open(restrictionsPath, "r")
            restrVector = restrFile.readline().split(',')
            restrFile.close()

            for idx, val in enumerate(restrVector):
                restrVector[idx] = int(val)
        except Exception as e:
            exit(f'ERR: restrictionsFile: {e}')
    k = 0
    # mapping domains to attributes in dictionary - this is a confusing calculation because of the
    # restrictions file - restrVector
    for idx, attr in enumerate(header):
        # drop attribute from consideration if number of unique vals is -1
        if domains[idx] != -1:
            if len(restrVector) != 0:
                if attr != classVar:
                    if restrVector[k] == 1:
                        attrDict[attr] = domains[idx]
                    else:
                        print("NOT CONSIDERING ATTRIBUTE: " + attr)
                    k += 1
                else:
                    attrDict[attr] = domains[idx]
            else:
                attrDict[attr] = domains[idx]
    return {'dataframe': trainingDf, 'domains': attrDict, 'classvar': classVar}