# Prints the command structure from the instructions and then exits, nice.
def printHelpMenuExit():
    print('\nUSAGE: python3 knn.py [csvFile] [NNearestNeighbors] <Verbose (True|False)> <Visualization (True|False)>')
    print('• csvFile: CSV datafile used for tree construction.')
    print('• NNearestNeighbors: how many neighbors each point should check to classify')
    print('• Verbose: DEFAULT True, show proccessing bar and other info')
    print('• Visualization: DEFAULT False, shows scatter plot comparisons between variables')
    exit()

def handleCommandLineParams(arguments):
    if not(len(arguments) == 3 or len(arguments) == 4 or len(arguments) == 5):
        print(f'Invalid number of arguments {len(arguments) - 1}, expected 2 or 3 or 4')
        printHelpMenuExit()
    else:
        dataPath = arguments[1]
        NNearestNeighbors = int(arguments[2])
        if len(arguments) == 4 or len(arguments) == 5: Verbose = bool(arguments[3])
        else: Verbose = True
        if len(arguments) == 5: Visualization = bool(arguments[4])
        else: Visualization = False
        return dataPath, NNearestNeighbors, Verbose, Visualization