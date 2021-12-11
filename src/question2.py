# Script Answering Question 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from kmeans import k_means, euclid_distances
from hclustering import agglomerative, get_clusters, get_leaf_nodes
from dbscan import DBSCAN, buildPointList, pointListToClusterList, centeroidnp


def createDotPlot(clusters, target, title):
    grade = (target / 20) * 100
    bin_ids = np.arange(0, 100, 2)
    subsets = []
    for cluster in np.unique(clusters):
        subset = grade[clusters == cluster]
        subsets.append(subset)
    plt.hist(subsets, bins=bin_ids, stacked=True)
    plt.title(title)
    plt.xlabel("Final Grades")
    plt.ylabel("Count")
    plt.show()


def createBarPlot(clusters, target, title):
    grade = target / 20
    labels = ["A", "B", "C", "D", "F"]
    assignments_total = [0, 0, 0, 0, 0]
    i = 0
    for cluster in np.unique(clusters):
        assignments = np.array([0, 0, 0, 0, 0])
        subset = grade[clusters == cluster]
        for point in subset:
            if point >= .9:
                assignments[0] += 1
            elif point >= .8:
                assignments[1] += 1
            elif point >= .7:
                assignments[2] += 1
            elif point >= .6:
                assignments[3] += 1
            else:
                assignments[4] += 1
        assignments_total[i] = assignments
        i += 1
    assignments_total = np.array(assignments_total)
    assignments_total = assignments_total / assignments_total.sum(axis=0)
    assignments_old = np.array([0, 0, 0, 0, 0], dtype='float64')
    for cluster in assignments_total:
        plt.bar(labels, cluster, width=.5, bottom=assignments_old)
        assignments_old += cluster
    plt.xlabel("Grade")
    plt.title(title)
    plt.show()


###################
## Control Panel ##
###################
target_name = "G3"
minMax = False  # Min / Max scale all columns
normal = False  # Normalize all columns
nothing = False  # True = don't drop anything | False = drop target_name

kmeans = True  # Run K-Means
heirac = False  # Run Hierarchical Clustering
dbscan = False  # Run DBSCANS

find_clusters = False  # True = hyper-parameter tuning | False = run / create graphs

k = 5  # Used by K-Means and Hierarchical Clustering
threshold = 25  # Used as starting for Hierarchical Clsutering
epsilon = 3.5  # Used by DBSCAN
numPoints = 4  # Used by DBSCAN


with open("../data/alcohol_dataset.pkl", "rb") as file:
    df = pickle.load(file)

df = df.reset_index(drop=True)    

target = df[target_name]

if nothing:
    data = df
else:
    # Drop only Target
    data = df.drop([target_name], axis=1)

# Min-Max Normalize all Columns 
if minMax:
    for column in data.columns.to_list():
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())

# Normalize all Columns 
elif normal:
    for column in data.columns.to_list():
        data[column] = (data[column] - data[column].mean()) / data[column].std()


if kmeans:
    if not find_clusters:
        kmeans_clusters = k_means(data, k)
        createDotPlot(kmeans_clusters, target, "Kmeans Clustering of Final Grades")
        createBarPlot(kmeans_clusters, target, "Kmeans Clustering of Final Grades")
    else:
        ks = []
        sses = []
        for k in range(2,30):
            ks.append(k)
            kmeans_clusters = k_means(data, k)
            clusters = pd.concat([data, pd.Series(kmeans_clusters, name="clusters")], axis=1)
            total_sse = 0
            print(f'K = {k}')
            for cluster in clusters.iloc[:, -1].unique():
                subset = clusters[clusters.iloc[:, -1] == cluster].drop(clusters.columns.to_list()[-1], axis=1)
                center = list(subset.mean())
                distances = euclid_distances(subset, [center])
                total_sse += (distances ** 2).sum()
            print(f'Total SSE = {round(float(total_sse), 3)}')
            sses.append(total_sse)
        plt.plot(ks, sses)
        plt.title("Total SSE by K Clusters (K-Means)")
        plt.xlabel("K")
        plt.xticks(np.arange(0, 30, 3))
        plt.ylabel("Total SSE")
        plt.show()

if heirac:
    dendrogram = agglomerative(data)
    if find_clusters:
        threshs = []
        sses = []
        while threshold > k:
            cluster_list = get_clusters(dendrogram, threshold)
            threshs.append(len(cluster_list)+1)
            assignments = pd.DataFrame()
            for cluster in range(len(cluster_list)):
                assignments = assignments.append(get_leaf_nodes(cluster_list[cluster], cluster), ignore_index=True)
            total_sse = 0
            for cluster in assignments.iloc[:, -1].unique():
                subset = assignments[assignments.iloc[:, -1] == cluster].drop(assignments.columns.to_list()[-1], axis=1)
                center = list(subset.mean())
                distances = euclid_distances(subset, [center])
                total_sse += (distances ** 2).sum().values[0]
            sses.append(total_sse)
            threshold -= 1
        plt.plot(threshs, sses)
        plt.title("Total SSE by Number of Clusters (Hierarchical Clustering)")
        plt.xlabel("Num Clusters")
        plt.ylabel("Total SSE")
        plt.show()

    else:
        cluster_list = get_clusters(dendrogram, threshold)
        while len(cluster_list) < 5:
            threshold -= .1
            cluster_list = get_clusters(dendrogram, threshold)
        print("Threshold:", threshold)
        print("Num Clusters:", len(cluster_list))
        hierac_clusters = pd.DataFrame()
        for cluster in range(len(cluster_list)):
            hierac_clusters = hierac_clusters.append(get_leaf_nodes(cluster_list[cluster], cluster), ignore_index=True)
        hierac_clusters.columns = ["Index", "Clusters"]
        hierac_clusters = hierac_clusters.set_index("Index")
        hierac_clusters = hierac_clusters.merge(target, how="inner", left_index=True, right_index=True)
        createDotPlot(hierac_clusters["Clusters"], hierac_clusters[target_name], "Agglomerative Clustering of Final Grades")
        createBarPlot(hierac_clusters["Clusters"], hierac_clusters[target_name], "Agglomerative Clustering of Final Grades")

if dbscan:
    if find_clusters:
        epsilon_range = np.arange(1, 1.02, .001)
        sses = []
        num_clus = []
        for epsilon in epsilon_range:
            listOfDBPoints = buildPointList(data, 0)
            numClusters = DBSCAN(listOfDBPoints, epsilon, 2, 0)
            num_clus.append(numClusters+1)
            print(f'Total Clusters {numClusters+1}')
            clustersArr = pointListToClusterList(listOfDBPoints, numClusters)
            total_sse = 0
            for cluster in clustersArr:
                cent = centeroidnp(cluster)
                total_sse += np.square(np.subtract(cluster, cent)).sum()
            print(f'Epsilon {epsilon}, SSE {round(total_sse,2)}')
            sses.append(total_sse)
        plt.plot(epsilon_range, sses)
        plt.xlabel("Epsilons")
        plt.ylabel("Total SSE")
        plt.title("Total SSE by Epsilon Value")
        plt.show()

        plt.plot(epsilon_range, num_clus)
        plt.xlabel("Epsilons")
        plt.ylabel("Number of Clusters")
        plt.title("Number of Clusters by Epsilon Value")
        plt.show()

    else:
        listOfDBPoints = buildPointList(data, 0)
        numClusters = DBSCAN(listOfDBPoints, epsilon, numPoints, 0)
        print("Num Clusters:", numClusters + 1)
        clustersArr = pointListToClusterList(listOfDBPoints, numClusters)
        combinedDf = pd.DataFrame()
        total_sse = 0
        for index, cluster in enumerate(clustersArr):
            if index != 0:
                print(cluster)
            cent = centeroidnp(cluster)
            total_sse += np.square(np.subtract(cluster, cent)).sum()
            combinedDf = pd.concat([combinedDf, pd.concat([pd.DataFrame(cluster), pd.Series(np.repeat(index, len(cluster)))], axis=1)])
        print(f'SSE = {round(total_sse, 2)}')
        combinedDf = combinedDf.reset_index(drop=True)
        columnNames = data.columns.to_list()
        columnNames.append("Clusters")
        combinedDf.columns = columnNames
        dbscan_clusters = combinedDf.merge(df, how="inner")
        createDotPlot(dbscan_clusters["Clusters"], dbscan_clusters[target_name], "DBSCANS Clustering of Final Grades")
        createBarPlot(dbscan_clusters["Clusters"], dbscan_clusters[target_name], "DBSCANS Clustering of Final Grades")
