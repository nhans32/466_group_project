import pandas as pd
import linecache

# Prints the command structure from the instructions and then exits, nice.
def printHelpMenuExit(opt=-1):
    if opt == 1:
        print('USAGE: python3 InduceC45 <trainingSetFile.csv> [<restrictionsFile>]')
    elif opt == 2:
        print('USAGE: python3 Classifier <CSVFile> <JSONFile>')
    elif opt == 3:
        pass
    else:
        print('Define an option for instructions.')
    exit()


# Custom command line handling for this program
def handleCommandLineParamsC45(arguments):
    # Handle basic argument restrictions, set trainingSetFile
    if len(arguments) == 1 or len(arguments) > 3:
        print(f'Invalid number of arguments {len(arguments) - 1}, expected 1|2')
        printHelpMenuExit(1)
    elif len(arguments) >= 2 and '.csv' not in arguments[1]:
        print(f'Not a valid trainingSetFile ({arguments[1]}) must be a csv.')
        printHelpMenuExit(1)
    trainingSetFile = arguments[1]
    # Do this part if there is possibly a restriction file
    if len(arguments) == 3 and '.txt' not in arguments[2]:
        print(f'Not a valid restrictionsFile ({arguments[2]}) must be a txt.')
        printHelpMenuExit()
    elif len(arguments) == 3:
        restrictionsFile = arguments[2]
    else:
        restrictionsFile = None
    # Return tuple with T/F if restriction file exists and list of files
    if restrictionsFile is None:
        isRestricted = False
    else:
        isRestricted = True
    return (isRestricted, [trainingSetFile, restrictionsFile])


# Custom command line handling for this program
def handleCommandLineParamsClassifier(arguments):
    # Handle basic argument restrictions, set trainingSetFile
    if len(arguments) != 3:
        print(f'Invalid number of arguments {len(arguments) - 1}, expected 2')
        printHelpMenuExit(2)
    if '.csv' not in arguments[1]:
        print(f'Not a valid CSVFile ({arguments[1]}) must be a csv.')
        printHelpMenuExit(2)
    # Do this part if there is possibly a restriction file
    if '.json' not in arguments[2]:
        print(f'Not a valid JSONFile ({arguments[2]}) must be a JSON.')
        printHelpMenuExit(2)
    CSVFile = arguments[1]
    JSONFile = arguments[2]
    # Return list with the two filenames
    return (CSVFile, JSONFile)


# for visualization of data - takes in dataframe dict
def printDFDict(dfDict):
    print("Domains: ", end='')
    print(dfDict['domains'])
    print("Class Variable: ", end='')
    print(dfDict['classvar'])
    print(dfDict['dataframe'])

# Read in one or two csvs based on what the user specified
def pandasReadFiles(parsedParamsTuple):
    try:
        trainingFilePath = parsedParamsTuple[1][0]

        # skipping rows 2 and 3 as they don't contain good info for dataFrame
        trainingDf = pd.read_csv(trainingFilePath, skiprows=[1, 2])  # pandas indexes differently than linecache

        header = linecache.getline(trainingFilePath, 1).strip().split(",")
        domains = list(map(int, linecache.getline(trainingFilePath, 2).strip().split(',')))
        classVar = linecache.getline(trainingFilePath, 3).strip()
        attrDict = {}

    except Exception as e:
        exit(f'ERR: trainingSetFile: {e}')

    restrVector = []
    # Only try to read the restrictions csv if we were provided one
    if parsedParamsTuple[0]:
        try:
            restrFile = open(parsedParamsTuple[1][1], "r")
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