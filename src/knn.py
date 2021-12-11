import pandas as pd
import numpy as np
import sys
from InduceC45 import read_csv
from time import time, sleep
from helpers.knnHelpers import *


def visualizeData(df, showFile, saveFile, filepath = None):
    # Doing imports here in order to reduce dependencies for non-visulization runs
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except:
        print("Install packages and try again. Skipping visualization.")
        sleep(2)
        return -1
    else:
        # Do the magic!
        plt.figure()
        sns.pairplot(df, hue = target_var)
        # Save and display as requested
        if saveFile and filepath is not None: plt.savefig(filepath)
        if showFile: plt.show()
        return 0

def calcNumericDistances(testRow, training_numeric, euclidean_distance):
    if euclidean_distance:
        distSubtr = testRow - training_numeric
        distSquares = distSubtr ** 2
        distSums = distSquares.sum(axis=1)
        distSqrts = distSums ** 0.5
        return distSqrts
    else:
        distSubtrAbs = np.abs(testRow - training_numeric)
        distSums = distSubtrAbs.sum(axis=1)
        return distSums

def calcCategoricalDistances(testRow, training_categorical):
    # get number of attributes
    num_attributes = len(testRow)
    count_matching = np.negative(np.count_nonzero(testRow == training_categorical, axis=1))
    distances = count_matching + num_attributes
    distances = distances / num_attributes
    return distances

def calcDistances(testRow_numeric, testRow_categorical, training_numeric, training_categorical, euclidean_distance):
    distances_numeric = None
    distances_categorical = None

    if np.size(training_numeric) > 0:
        distances_numeric = calcNumericDistances(testRow_numeric, training_numeric, euclidean_distance)
    if np.size(training_categorical) > 0:
        distances_categorical = calcCategoricalDistances(testRow_categorical, training_categorical)

    if distances_numeric is None:
        distances = distances_categorical
    elif distances_categorical is None:
        distances = distances_numeric
    else:
        distances = np.add(distances_numeric, distances_categorical)
    return distances


def knn(k, testRow, idx, df, df_np, df_numeric, df_categorical, target_var, target_domain, euclidean_distance):
        training_total = np.concatenate([df_np[:idx], df_np[idx + 1:]])
        training_numeric = np.concatenate([df_numeric[:idx], df_numeric[idx + 1:]])
        training_categorical = np.concatenate([df_categorical[:idx], df_categorical[idx + 1:]])

        testRow_numeric = df_numeric[idx]
        testRow_categorical = df_categorical[idx]

        distances = calcDistances(testRow_numeric, testRow_categorical, training_numeric, training_categorical, euclidean_distance)

        # # get first k smallest distances based on value in distances
        first_k_smallest_dist = np.argsort(distances)[:k]

        poss_predictions = [df.iloc[small_dist][target_var] for small_dist in first_k_smallest_dist]

        prediction = max(set(poss_predictions), key=poss_predictions.count)

        return prediction


def runknn(df, k, numeric_attributes, categorical_attributes, target_var, target_domain, euclidean_distance, verbose):
    # convert df into numpy array
    df_np = df.to_numpy()
    # get dataframe with only numeric attributes columns and convert to numpy array
    df_numeric = df[numeric_attributes].to_numpy()
    # get dataframe with only categorical attributes columns and convert to numpy array
    df_categorical = df[categorical_attributes].to_numpy()

    correct_predictions = 0
    incorrect_predictions = 0
    predictions = 0

    if verbose: print(f'Classifying {len(df_np)} entries. Each █ represents 10 entries.')
    predict_dict = dict()
    predict_dict2 = dict()
    for idx, testRow in enumerate(df_np):
        # this could be less than efficient
        # get array of values before and after testRow
        prediction = knn(k, testRow, idx, df, df_np, df_numeric, df_categorical, target_var, target_domain, euclidean_distance)
        if prediction not in predict_dict: predict_dict[prediction] = dict()
        if prediction not in predict_dict2: predict_dict2[prediction] = 0
        predict_dict2[prediction] += 1
        # compare prediction to target_var value of testRow
        predictions += 1
        if prediction == df.iloc[idx][target_var]: correct_predictions += 1
        else: incorrect_predictions += 1
        if df.iloc[idx][target_var] not in predict_dict[prediction]: predict_dict[prediction][df.iloc[idx][target_var]] = 0
        predict_dict[prediction][df.iloc[idx][target_var]] += 1
        if verbose and predictions % 10 == 0: print('█', end ='', flush = True)
        if verbose and predictions % 1000 == 0: print(f' DONE {predictions}')
    if verbose: print(f' DONE {predictions}')
    print("Correct Predictions: ", correct_predictions)
    print("Incorrect Predictions: ", incorrect_predictions)
    print("Total Predictions: ", predictions)
    print("Accuracy: ", correct_predictions / predictions)
    print("Error Rate: ", incorrect_predictions / predictions)
    print(predict_dict2)


if __name__ == '__main__':
    file_name, k, verbose, visulization = handleCommandLineParams(sys.argv)
    df, attributes, attribute_domains, target_var, target_domain = read_csv(file_name, "")

    # Calling visualize here if desired
    # ********* MUST INSTALL ********* matplotlib and seaborn
    if visulization:
        f = file_name.strip('.csv').split('/')
        outputImageName = 'question4_out/knnVisualize/' + f[len(f)-1] + '.png'
        visualizeData(df, False, True, outputImageName)

    numeric_attributes = []
    categorical_attributes = []
    # normalize numeric attributes to 0-1 and get list of numerical attributes
    for idx, attr in enumerate(attributes):
        if attribute_domains[idx] == 0:
            numeric_attributes.append(attr)
            # reindex dataframe
            df = df.reset_index(drop=True)
            df[attr] = (df[attr] - df[attr].min()) / (df[attr].max() - df[attr].min())
        else:
            categorical_attributes.append(attr)

    runknn(df, k, numeric_attributes, categorical_attributes, target_var, target_domain, True, verbose)
