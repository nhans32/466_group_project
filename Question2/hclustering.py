import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from kmeans import euclid_distances
import linecache
import sys

# For encoding the JSON
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def distance_matrix(data):
    distances = pd.DataFrame()
    for row in range(len(data)):
        distances[row] = np.sqrt(((data - data.iloc[row]) ** 2).sum(axis=1))
    np.fill_diagonal(distances.values, np.NAN)
    return distances


def agglomerative(data):
    # Initialize Clusters, Leaf Nodes, and Distance Matrix
    clusters = [[x] for x in range(len(data))]
    #nodes = {x: {"type": "leaf", "height": 0.0, "data": list(data.iloc[x].values)} for x in range(len(data))}
    nodes = {x: {"type": "leaf", "height": 0.0, "data": x} for x in range(len(data))}
    distances = distance_matrix(data)

    # Size keeps track of original data size; i is how many new clusters - 1
    size = len(clusters)
    i = 0

    while len(clusters) < 2 * size - 1:
        # Find smallest distance (dist), r = rowIndex, c = colIndex
        r, c, dist = distances.loc[:, distances.min().idxmin()].idxmin(), distances.min().idxmin(), distances.min().min()

        # Create the new clusters and nodes
        clusters.append([r,c])
        if len(clusters) == 2 * size - 1:
            nodes[size + i] = {"type": "root", "height": dist, "nodes": [nodes[r], nodes[c]]}
        else:
            nodes[size + i] = {"type": "node", "height": dist, "nodes": [nodes[r], nodes[c]]}

        # Add the complete link distance for new cluster, and delete old clusters
        distances[size+i] = distances[[r, c]].max(axis=1)
        distances = distances.append(pd.Series(distances[[r, c]].max(axis=1), name=size+i))
        distances = distances.drop([r, c], axis=1)
        distances = distances.drop([r, c])

        i += 1
    return nodes[size * 2 - 2]

# Gets a list where each element is a cluster
def get_clusters(tree, threshold):
    clusters = []
    if tree["height"] < threshold:
      return [tree]
    left = get_clusters(tree["nodes"][0], threshold)
    right = get_clusters(tree["nodes"][1], threshold)
    for cluster in left:
        clusters.append(cluster)
    for cluster in right:
        clusters.append(cluster)
    return clusters  

# Convert individual clusters in dataframe of data with assignments as last column (left)
def get_leaf_nodes(tree, cluster):
    clusters = []
    if tree["type"] == "leaf":
        #values = list(tree["data"])
        values = [tree["data"]]
        values.append(cluster)
        return [values]
    left = get_leaf_nodes(tree["nodes"][0], cluster)
    right = get_leaf_nodes(tree["nodes"][1], cluster)
    for cluster in left:
        clusters.append(cluster)
    for cluster in right:
        clusters.append(cluster)
    return clusters

def exitHelpMessage(error = None):
    if error is not None: print(error)
    print('USAGE: python hclustering.py <Filename> <k>')
    print('- <Filename> is the name of the CSV file containing the input dataset.')
    print('- [<threshold>] is the optional number of clusters the program has to produce.')
    exit(-1)

def handleCommandLineParams(arguments):
    if len(arguments) == 2:
        fileName = arguments[1]
        threshold = None
    elif len(arguments) == 3: 
        fileName = arguments[1]
        try: threshold = int(arguments[2])
        except: exitHelpMessage("Argument threshold is not an int")
    else: exitHelpMessage("Argument count incorrect")
    return fileName, threshold

def readData(fileName, dropcols = []):
    try: df = pd.read_csv(fileName, skiprows = 1, header = None)
    except Exception as e: exit(f"ERROR ON {fileName}: {e}")
    header = linecache.getline(fileName, 1).strip().split(",")
    # Drop columns for which the header has a value of '0'
    for i, col in enumerate(header):
        if col == '0': dropcols.append(i)
    df = df.drop(dropcols, axis=1)
    return df

def create_output(data):
    PRECISION = 2
    np.set_printoptions(formatter={'float': f'{{:0.{PRECISION}f}}'.format})
    print ('------------------------------------')
    print(f'-- HIERARCHICAL CLUSTERING OUTPUT --')
    print ('------------------------------------')
    for cluster in data.iloc[:, -1].unique():
        subset = data[data.iloc[:, -1] == cluster].drop(data.columns.to_list()[-1], axis=1)
        center = list(subset.mean())
        distances = euclid_distances(subset, [center])
        SSE = (distances ** 2).sum()
        print(f'------------ Cluster: {cluster} ------------')
        print(f'Center: {center}')
        print(f'Max Distance to Center: {round(float(np.max(distances)), PRECISION)}')
        print(f'Min Distance to Center: {round(float(np.min(distances)), PRECISION)}')
        print(f'Avg Distance to Center: {round(float(np.mean(distances)), PRECISION)}')
        print(f'Sum of Squared Errors: {round(float(SSE), PRECISION)}')
        # 6. Sum of Squared Errors (SSE) for the points in the cluster.
        print(f'{len(subset)} Points:')
        for point in range(len(subset)): print(list(subset.iloc[point]))
        print ('------------------------------------')

if __name__ == "__main__":
    TESTING = False
    if TESTING:
        fileName = "input_files/many_clusters.csv"
        threshold = 28
    else: fileName, threshold = handleCommandLineParams(sys.argv)
    data = readData(fileName)

    dendrogram = agglomerative(data)
    print(json.dumps(dendrogram, indent=1, cls=NpEncoder))

    if threshold:
        cluster_list = get_clusters(dendrogram, threshold)
        assignments = pd.DataFrame()
        for cluster in range(len(cluster_list)):
            assignments = assignments.append(get_leaf_nodes(cluster_list[cluster], cluster), ignore_index=True)

        create_output(assignments)

    #plt.scatter(assignments.iloc[:,1], assignments.iloc[:,2], c=assignments.iloc[:,-1])
    #plt.show()
