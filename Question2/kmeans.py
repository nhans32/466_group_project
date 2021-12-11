import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import linecache
import sys

def euclid_distances(X, centroids):
    distances = pd.DataFrame()
    for cluster in range(len(centroids)):
        distances[cluster] = np.sqrt(((X - centroids[cluster]) ** 2).sum(axis=1))
    return distances

def select_Centroids(data, k):
    X = data.copy()
    data_center = X.mean()
    centroids = [[] for i in range(k)]
    index = int(euclid_distances(X, [data_center]).idxmax().values)
    centroids[0] = list(X.loc[index])
    X = X.drop(index)
    for cluster in range(1, k):
        index = euclid_distances(X, centroids[:cluster]).sum(axis=1).argmax()
        centroids[cluster] = list(X.iloc[index])
        X = X.reset_index(drop=True).drop(index)
    return centroids

def k_means(data, k):
    X = data.copy()
    centroids = select_Centroids(X, k)
    epochs = 1
    while epochs <= 500:
        clusters = euclid_distances(X, centroids).idxmin(axis=1).values
        old_centroids = centroids.copy()
        for centroid in range(len(centroids)):
            centroids[centroid] = np.array(X.iloc[clusters == centroid].mean())
        if np.equal(centroids, old_centroids).all():
            break
    return clusters

def exitHelpMessage(error = None):
    if error is not None: print(error)
    print('USAGE: python kmeans.py <Filename> <k>')
    print('- <Filename> is the name of the CSV file containing the input dataset.')
    print('- <k> is the number of clusters the program has to produce.')
    exit(-1)

def handleCommandLineParams(arguments):
    if len(arguments) == 3:
        fileName = arguments[1]
        try: k = int(arguments[2])
        except: exitHelpMessage("Argument k is not an int")
    else: exitHelpMessage("Argument count incorrect")
    return fileName, k

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
    print(f'----- KMEANS CLUSTERING OUTPUT -----')
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
        fileName = "input_files/mammal_milk.csv"
        k = 4
    else: fileName, k = handleCommandLineParams(sys.argv)
    data = readData(fileName)

    k_means_result = k_means(data, k)

    clusters = pd.concat([data, pd.Series(k_means_result, name="clusters")], axis=1)

    create_output(clusters)

    #plt.scatter(clusters.iloc[:,0], clusters.iloc[:,1], c=clusters.clusters)
    #plt.show()
