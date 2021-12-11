import pandas as pd
import linecache
import json
from dataStructs.Node import Node
from dataStructs.Edge import Edge

# Prints the command structure from the instructions and then exits, nice.
def printHelpMenuExit():
    print('USAGE: python3 classifier.py [csvFile] [JSONFile]')
    exit()

def handleCommandLineParams(arguments):
    if len(arguments) != 3:
        print(f'Invalid number of arguments {len(arguments) - 1}, expected 2')
        printHelpMenuExit()
    else:
        return (pandasParseCSV(arguments[1]), parseJSONTree(arguments[2]))

def pandasParseCSV(csvFile):
    # skipping rows 2 and 3 as they don't contain good info for dataFrame
    try:
        trainingSet = True

        classifyDf = pd.read_csv(csvFile, skiprows=[1, 2])  # pandas indexes differently than linecache

        header = linecache.getline(csvFile, 1).strip().split(",")
        # strip quotes from header
        header = [x.strip('"') for x in header]
        domains = linecache.getline(csvFile, 2).strip().split(",")
        # strip quotes from domains
        domains = [x.strip('"') for x in domains]
        # convert domains to int
        domains = [int(x) for x in domains]
        classVar = linecache.getline(csvFile, 3).strip()
        classVar = classVar.strip('"')

        if classVar == "":
            trainingSet = False

        # if attribute domain is -1, then remove it from attributes and attribute domains
        for idx, attribute in enumerate(header):
            if domains[idx] == -1:
                header.remove(attribute)
                domains.remove(domains[idx])
    except Exception as e:
        exit(f'ERR: trainingSetFile: {e}')

    return ((classifyDf, header, domains, classVar), trainingSet)


def parseJSONTree(JSONFile):
    jsonF = open(JSONFile)
    tree = json.load(jsonF)
    jsonF.close()
    return parseJsonToTree(tree)

def parseJsonToTree(data):
    if 'leaf' in data:
        newLeaf = Node(data['leaf']['decision'], True)
        newLeaf.p = data['leaf']['p']
        return newLeaf
    elif 'node' in data:
        newNode = Node(data['node']['var'], False)
        newNode.children = []
        for edge in data['node']['edges']:
            newEdge = Edge(edge['edge']['value'], None, None)
            if edge['edge']['value'] == None:
                newEdge.ghost = True
            else:
                newEdge.ghost = False
            if 'direction' in edge['edge']:
                newEdge.continuous = True
                newEdge.direction = edge['edge']['direction']
            else:
                newEdge.continuous = False
            newEdge.endNode = parseJsonToTree(edge['edge'])
            newNode.children.append(newEdge)
        return newNode
