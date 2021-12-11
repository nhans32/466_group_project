# Prints the command structure from the instructions and then exits, nice.
def printHelpMenuExit():
    print('\nUSAGE: python3 randomForest.py [csvFile] [NumAttributes] [NumDataPoints] [NumTrees]')
    print('• csvFile: CSV datafile used for tree construction.')
    print('• NumAttributes: this parameter controls how many attributes \n\
    each decision tree built by the Random Forest classifier shall contain')
    print('• NumDataPoints: the number of data points selected randomly \n\
    with replacement to form a dataset for each decision tree.')
    print('• NumTrees: the number of the decision trees to build.\n')
    exit()

def handleCommandLineParams(arguments):
    if not len(arguments) == 5:
        print(f'Invalid number of arguments {len(arguments) - 1}, expected 4')
        printHelpMenuExit()
    else:
        dataPath = arguments[1]
        NumAttributes = int(arguments[2])
        NumDataPoints = int(arguments[3])
        NumTrees = int(arguments[4])
        return dataPath, NumAttributes, NumDataPoints, NumTrees