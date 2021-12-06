# Script Answering Question 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from kmeans import k_means
from hclustering import agglomerative, get_clusters, get_leaf_nodes
from dbscan import DBSCAN, buildPointList, pointListToClusterList

def createDotPlot(clusters, target, title, target_name):
    if target_name == "G3":
        grade = (target / 20) * 100
        bin_ids = np.arange(0, 100, 2)
    elif target_name == "school":
        grade = target
        bin_ids = [0, .5, 1]
    subsets = []
    for cluster in np.unique(clusters):
        subset = grade[clusters == cluster]
        subsets.append(subset)
    plt.hist(subsets, bins=bin_ids, stacked=True)
    plt.title(title)
    plt.xlabel("Final Grades")
    plt.ylabel("Count")
    plt.show()


    


with open("data/alcohol_dataset.pkl", "rb") as file:
    df = pickle.load(file)

noGrades = False
minMax = False
normal = False
target_name = "G3"


df = df.reset_index(drop=True)    

target = df[target_name]

if noGrades:
    # Drop all Grades
    data = df.drop(["G1", "G2", "G3"], axis=1)
else:
    # Drop only Target
    data = df.drop([target_name], axis=1)

###################################
## Min-Max Normalize all Columns ##
###################################
if minMax:
    for column in data.columns.to_list():
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
#############################
## Standardize all Columns ##
#############################
elif normal:
    for column in data.columns.to_list():
        data[column] = (data[column] - data[column].mean()) / data[column].std()



kmeans = True
heirac = False
dbscan = False


if kmeans:
    k = 5
    kmeans_clusters = k_means(data, k)
    createDotPlot(kmeans_clusters, target, "Kmeans Clustering of Final Grades", target_name)

if heirac:
    k = 2
    threshold = 80
    dendrogram = agglomerative(data)
    cluster_list = get_clusters(dendrogram, threshold)
    while len(cluster_list) < k:
        threshold -= 5
        cluster_list = get_clusters(dendrogram, threshold)
    print("Threshold:", threshold)
    hierac_clusters = pd.DataFrame()
    for cluster in range(len(cluster_list)):
        hierac_clusters = hierac_clusters.append(get_leaf_nodes(cluster_list[cluster], cluster), ignore_index=True)
    hierac_clusters.columns = ["Index", "Clusters"]
    hierac_clusters = hierac_clusters.set_index("Index")
    hierac_clusters = hierac_clusters.merge(target, how="inner", left_index=True, right_index=True)
    createDotPlot(hierac_clusters["Clusters"], hierac_clusters[target_name], "Agglomerative Clustering of Final Grades", target_name)

if dbscan:
    epsilon = 6
    numPoints = 6
    listOfDBPoints = buildPointList(data, 0)
    numClusters = DBSCAN(listOfDBPoints, epsilon, numPoints, 0)
    print("Num Clusters:", numClusters + 1)
    clustersArr = pointListToClusterList(listOfDBPoints, numClusters)
    combinedDf = pd.DataFrame()
    for index, cluster in enumerate(clustersArr):
        combinedDf = pd.concat([combinedDf, pd.concat([pd.DataFrame(cluster), pd.Series(np.repeat(index, len(cluster)))], axis=1)])
    combinedDf = combinedDf.reset_index(drop=True)
    columnNames = data.columns.to_list()
    columnNames.append("Clusters")
    combinedDf.columns = columnNames
    dbscan_clusters = combinedDf.merge(df, how="inner")
    createDotPlot(dbscan_clusters["Clusters"], dbscan_clusters[target_name], "DBSCANS Clustering of Final Grades", target_name)

