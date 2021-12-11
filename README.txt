PYTHON VERSION:
#   3.9.7
# important to maintain dictionary ordering

REQUIRED PACKAGES:
itertools
collections
optparse
pandas
sys
matplotlib
numpy
sklearn

466 Final Group Project
Authors:
Nick Hansen - nhanse02@calpoly.edu
Otakar Andrysek - oandryse@calpoly.edu
Nathan Johnson - njohns60@calpoly.edu
Edward Zhou - ekzhou@calpoy.edu

Description:
All scripts are located in the /src folder.
The original dataset from Kaggle is student-mat.csv and student-por.csv in /data, which corresponds to the math and Portuguese classes.
The merged and preprocessed dataset is alcohol_dataset.pkl in /data.
Scripts containing code to generate statistics/ discover insights for each question is labeled questionX.py in the /src folder, where X is an integer corresponding to the question number.
Parameters can be modified in the main functions.

===================
For Question 1:
Outputs from question1.py located in /outputs/.
Outputs from q1_analysis.py located in /analysis_outputs/.

===================
For Question 2:
The control panel starts at line 59, which is used to change parameters for clustering.

Control panel options:
- minMax : if true it min/max scales all columns of the dataset
- norm : if true normalizes all columns of the dataset
- nothing : if true doesn't drop the target column

- kmeans : if true runs K-Means algorithm
- heirac : if true runs Hierarcichal Clustering
- dbscan : if true runs the DBSAN algorithm

- find_clusters : if true does parameter tuning and creates graph of Total SSE

===================
For Question 3:
Commented out code was used to generate statistics, but is not run for simplicity's sake.

===================
For Question 4:
Input: 466_group_project/data/[student-mat.csv or student-por.csv]
Output: 466_group_project/src/question4_out/
- knn: run knn.py with no options and list of paramters will be returned
- c45: the paramters used for the experiment are set. Can be modified in the main function near end of the python file.
- randomForest: run randomForest.py, the paramters used for the experiment are set. Can be modified in the main function near end of the python file.
