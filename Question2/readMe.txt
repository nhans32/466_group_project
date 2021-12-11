Essentially the code is quite ugly, but I made a control panel that can control everything

The conrol panel starts at line 59

Below is a more detailed explanation of each control panel option:
- minMax : if true it min/max scales all columns of the dataset
- norm : if true normalizes all columns of the dataset
- nothing : if true doesn't drop the target column

- kmeans : if true runs K-Means algorithm
- heirac : if true runs Hierarcichal Clustering
- dbscan : if true runs the DBSAN algorithm

- find_clusters : if true does parameter tuning and creates graph of Total SSE 