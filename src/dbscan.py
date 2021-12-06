import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import linecache
import sys
import os


class DBPoint:
    def __init__(self, location):
        # Coordinates of the point, as a list
        self.location = location
        # Dimmension of the point (how many coodinates are associated with it)
        self.dim = len(location)
        # DBSCAN type for the point can be core [2], boundry [1], or outlier [0]
        self.type = None
        # The cluster identifier for the point, as an int
        self.cluster = None
        # Boolean T/F for if the point has been visited
        self.visited = False
        # Number of points in the cluter
        self.numNeighbors = 0
        # List of DBPoints contained in this points cluster
        self.neighbors = []

    def __repr__(self):
        return f'DBPoint: {self.location} (dim: {self.dim}); type: {self.type}; cluster: {self.cluster}; visited: {self.visited}; numNeighbors: {self.numNeighbors}; neighbors: not shown'

def exitHelpMessage(error = None):
    if error is not None: print(error)
    print('USAGE: python dbscan <Filename> <epsilon> <NumPoints>')
    print('- <Filename> is the name of the CSV file containing the input dataset.')
    print('- <epsilon> is the Epsilon (ε) parameter of the DBSCAN algorithm: \n\
            the radius within which DBSCAN shall search for points.')
    print('- <NumPoint> is the minimal number of points within the <epsilon> \n\
             distance from a given point to continue building the cluster')
    exit(-1)

def handleCommandLineParams(arguments):
    if len(arguments) == 4:
        fileName = arguments[1]
        try: epsilon = int(arguments[2])
        except: exitHelpMessage("epsilon argument is not int")
        try: numPoints = int(arguments[3])
        except: exitHelpMessage("NumPoint argument is not int")
    else: exitHelpMessage("Argument count incorrect")
    return fileName, epsilon, numPoints

def readData(fileName, dropcols = []):
    try: df = pd.read_csv(fileName, skiprows = 1, header = None)
    except Exception as e: exit(f"ERROR ON {fileName}: {e}")
    header = linecache.getline(fileName, 1).strip().split(",")
    # Drop columns for which the header has a value of '0'
    for i, col in enumerate(header):
        if col == '0': dropcols.append(i)
    df = df.drop(dropcols, axis=1)
    return df

def buildPointList(data, VERBOSE):
    listOfDBPoints= []
    # This is a slow operation, improve if effeciency is a concern
    for point in data.itertuples():
        coordinates = list(point)
        P = DBPoint(np.asarray(coordinates[1:]))
        listOfDBPoints.append(P)
    if VERBOSE:
        for point in listOfDBPoints: print(point)
    return listOfDBPoints

def DBSCAN(listOfDBPoints, epsilon, numPoints, VERBOSE):
    for point in listOfDBPoints:
        # If we have visited this point already don't check it again
        if point.visited: continue
        for point2 in listOfDBPoints:
            # Check if the point is in range
            point2InRange = pointInEpsilon(point.location, point2.location, point.dim, epsilon)

            # If it is in range, append it to the the first points neighbors
            if point2InRange: 
                if VERBOSE > 1: print(f"FOUND {point2.location} in range of {point.location}")
                point.neighbors.append(point2)
                point.numNeighbors += 1

        #Mark the point as visited
        point.visited = True
        # If it has enough neighbors it is a core point
        if point.numNeighbors >= numPoints: point.type = 2
            
    # We now have all points which are core, each in their own cluster
    curCluster = 0
    for point in listOfDBPoints:
        if not point.type == 2: continue
        if point.cluster is None: 
            point.cluster = curCluster
            expandCluster(point, listOfDBPoints, curCluster)
            curCluster += 1
    for point in listOfDBPoints:
        if point.type is None:
            if point.cluster is not None: point.type = 1
            else: 
                point.cluster =-1
                point.type = 0
    if VERBOSE: 
        for point in listOfDBPoints: print(point)
    return curCluster


def expandCluster(point, listOfDBPoints, curCluster):
    for pointNeighbor in point.neighbors:
        if pointNeighbor.type == 2 and pointNeighbor.cluster != curCluster:
            pointNeighbor.cluster = curCluster
            expandCluster(pointNeighbor, listOfDBPoints, curCluster)
        elif pointNeighbor.cluster != curCluster: pointNeighbor.cluster = curCluster
        

def pointInEpsilon(p1, p2, dim, epsilon = None):
    distance = np.linalg.norm(p1 - p2)
    if epsilon is not None: return (distance <= epsilon)
    return distance


def pointListToClusterList(listOfDBPoints, numClusters):
    clustersArr = []
    for curCluster in range(-1, numClusters):
        cluster = ([point.location for point in listOfDBPoints if point.cluster == curCluster])
        clustersArr.append(np.asarray(cluster))
    return clustersArr


def centeroidnp(arr):
    length, dim = arr.shape
    return np.array([np.sum(arr[:, i])/length for i in range(dim)])


def outputResults(listOfDBPoints, numClusters):
    clustersArr = pointListToClusterList(listOfDBPoints, numClusters)
    dim = listOfDBPoints[0].dim
    PRECISION = 2
    np.set_printoptions(formatter={'float': f'{{:0.{PRECISION}f}}'.format})

    print ('------------------------------------')
    print(f'----- DBSCAN CLUSTERING OUTPUT -----')
    print ('------------------------------------')

    for index, cluster in enumerate(clustersArr):
        if index == 0: continue   # Skip the outlier cluster, it will be printed at the end
        cent = centeroidnp(cluster)
        distances = [pointInEpsilon(point, cent, dim) for point in cluster]
        print(f'------------ Cluster: {index} ------------')
        print(f'Center: {cent}')
        print(f'Max Distance to Center: {round(np.max(distances), PRECISION)}')
        print(f'Min Distance to Center: {round(np.min(distances), PRECISION)}')
        print(f'Avg Distance to Center: {round(np.mean(distances), PRECISION)}')
        print(f'Sum of Squared Errors: {round(np.square(np.subtract(cluster, cent)).sum(), PRECISION)}')
        # 6. Sum of Squared Errors (SSE) for the points in the cluster.
        print(f'{len(cluster)} Points:')
        for point in cluster: print(point)
        print ('------------------------------------')
    # Outlier info
    print(f'-------- OUTLIER STATISTICS --------')
    print(f'Percent of data as outliers: {round(len(clustersArr[0])/len(listOfDBPoints), PRECISION) * 100}%')
    print(f'Total number of outliers: {len(clustersArr[0])}')
    for point in clustersArr[0]: print(point)


if __name__ == "__main__":
    TESTING = False
    VERBOSE = 0
    SHOW_PLOT = False
    if TESTING:
        fileName = "input_files/many_clusters.csv"
        epsilon = 6
        numPoints = 6
    else: fileName, epsilon, numPoints = handleCommandLineParams(sys.argv)
    data = readData(fileName)
    listOfDBPoints = buildPointList(data, VERBOSE)
    numClusters = DBSCAN(listOfDBPoints, epsilon, numPoints, VERBOSE)

    print(f'Epsilon: {epsilon}, MinPoints: {numPoints} - Created {numClusters} clusters.')
    outputResults(listOfDBPoints, numClusters)


    if SHOW_PLOT:
        clusters = []
        for point in listOfDBPoints: clusters.append(point.cluster)
        clusters = pd.concat([data, pd.Series(clusters, name="clusters")], axis=1)

        f = fileName.replace('.csv', '.png').split('/')
        outputImageName = 'out/dbscan/' + f[len(f)-1]
        
        plt.scatter(clusters.iloc[:,0], clusters.iloc[:,1], c=clusters.clusters)
        
        plt.savefig(outputImageName)
        plt.show()
        