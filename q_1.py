import pandas as pd
import numpy as np
import itertools
import re
import sys

sys.path.insert(0, 'Apriori-python3/')
from apriori import *

# hard coded bins for various variables (ie whose continous vars have too many unique values to be raw one hot encoded)
def createBins(df):
    # vars_to_bin = ['absences', 'G1', 'G2', 'G3']
    print('creating Bins...')

    min_absences = df['absences'].min()
    max_absences = df['absences'].max()
    print('min absences: ', min_absences)
    print('max absences: ', max_absences)
    bins_absences = [i for i in range(min_absences, max_absences + 5, 5)]
    df['binned_absences'] = pd.cut(df['absences'], bins=bins_absences, include_lowest=True)
    print(f'absences bins: {bins_absences}')
    print(df['binned_absences'].value_counts())
    # cleaning up the bin labels
    df['binned_absences'] = df['binned_absences'].apply(str).str.replace(', ', '_').str.replace('(', '').str.replace(']', '')
    df = df.drop('absences', axis=1)

    min_G1 = df['G1'].min()
    max_G1 = df['G1'].max()
    print('G1 Min:', min_G1)
    print('G1 Max:', max_G1)
    bins_G1 = [i for i in range(min_G1, max_G1 + 5, 5)]
    df['binned_G1'] = pd.cut(df['G1'], bins=bins_G1, include_lowest=True)
    print(f'G1 bins: {bins_G1}')
    print(df['binned_G1'].value_counts())
    df['binned_G1'] = df['binned_G1'].apply(str).str.replace(', ', '_').str.replace('(', '').str.replace(']', '')
    df = df.drop('G1', axis=1)

    min_G2 = df['G2'].min()
    max_G2 = df['G2'].max()
    print('G2 Min:', min_G2)
    print('G2 Max:', max_G2)
    bins_G2 = [i for i in range(min_G2, max_G2 + 5, 5)]
    df['binned_G2'] = pd.cut(df['G2'], bins=bins_G2, include_lowest=True)
    print(f'G2 bins: {bins_G2}')
    print(df['binned_G2'].value_counts())
    df['binned_G2'] = df['binned_G2'].apply(str).str.replace(', ', '_').str.replace('(', '').str.replace(']', '')
    df = df.drop('G2', axis=1)

    min_G3 = df['G3'].min()
    max_G3 = df['G3'].max()
    print('G3 Min:', min_G3)
    print('G3 Max:', max_G3)
    bins_G3 = [i for i in range(min_G3, max_G3 + 5, 5)]
    df['binned_G3'] = pd.cut(df['G3'], bins=bins_G3, include_lowest=True)
    print(f'G3 bins: {bins_G3}')
    print(df['binned_G3'].value_counts())
    df['binned_G3'] = df['binned_G3'].apply(str).str.replace(', ', '_').str.replace('(', '').str.replace(']', '')
    df = df.drop('G3', axis=1)

    df.rename(columns={'binned_absences': 'absences', 'binned_G1': 'G1', 'binned_G2': 'G2', 'binned_G3': 'G3'}, inplace=True)

    # df = oneHotEncode(df, ['absences', 'G1', 'G2', 'G3'])
    return df

def pruneAndOutputDF(df, output_file):
    with open(output_file, 'w') as f:
        # iterate over rows of dataframe
        for index, row in df.iterrows():
            # if a column has a value of 1 then write that column name to the output file
            for idx, col in enumerate(row.index):
                if row[col] == 1:
                    f.write(f'{col},')
                if idx == len(row.index) - 1:
                    f.write('\n')
        f.close()

# further hot encoding for the alcohol dataset
def oneHotEncode(df, one_hot_vars):
    for var in one_hot_vars:
        one_hot = pd.get_dummies(df[var])
        one_hot_new_columns = [str(var) + '_' + str(label) for label in list(one_hot.columns)]
        one_hot.columns = one_hot_new_columns

        df = df.drop(var, axis=1)
        df = df.join(one_hot)

    return df

def outputItemsRules(items, rules, min_supp, min_conf, dataset, prune):
    # open output file outputs/apriori_items_min_supp_0.1_min_conf_0.5.txt
    items_sort = sorted(items, key=lambda x: x[1], reverse=True)
    with open(f'outputs/apriori_items_supp_{min_supp}_{dataset}_prune_{prune}.txt', 'w') as f:
        for item in items_sort:
            f.write(f'{item[0]} with support: {item[1]}\n')
        f.close()
    
    rules_sort = sorted(rules, key=lambda x: x[1], reverse=True)
    with open(f'outputs/apriori_rules_supp_{min_supp}_conf_{min_conf}_{dataset}_prune_{prune}.txt', 'w') as f:
        for rule in rules_sort:
            f.write(f'{rule[0][0]} ====> {rule[0][1]} with confidence: {rule[1]}\n')
        f.close()

if __name__ == '__main__':
    MIN_SUPP = 0.45
    MIN_CONF = 0.85
    # can be 'combined', 'mat' or 'por'
    DATASET = 'combined'
    PRUNE = True

    hot_encoding_vars = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
                         'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
                         'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']

    if DATASET == 'mat':
        df_main = pd.read_csv('data/student-mat.csv')
    elif DATASET == 'por':
        df_main = pd.read_csv('data/student-por.csv')
    elif DATASET == 'combined':
        df_mat = pd.read_csv('data/student-mat.csv')
        df_mat['class_type'] = 'mat'
        df_por = pd.read_csv('data/student-por.csv')
        df_por['class_type'] = 'por'
        df_combined = pd.concat([df_mat, df_por], axis=0)
        # reset index of combined dataframe, if don't do this - will run out of memory FASSSTTT
        df_combined = df_combined.reset_index(drop=True)
        # account for class type
        hot_encoding_vars.append('class_type')
        df_main = df_combined
        print(hot_encoding_vars)

    # discretizing the continous variables and final hot encoding of these variables
    df_main = createBins(df_main)

    if PRUNE:
        # percentage of dataframe the most freq value in column takes up
        top_val_counts = {}
        for var in hot_encoding_vars:
            top_val_counts[var] = list(df_main[var].value_counts())[0]/len(df_main)

        # prune off columns with most freq value making up >=85% of dataframe
        # if top_val_counts[var] >= 0.85: remove var from hot_encoding_vars and df_main
        for key, val in top_val_counts.items():
            if val >= 0.85:
                hot_encoding_vars.remove(key)
                df_main = df_main.drop(key, axis=1)

    # one hot encoding for the rest of the rows
    df_main = oneHotEncode(df_main, hot_encoding_vars)

    pruneAndOutputDF(df_main, f'data/alcohol_dataset_apriori_{DATASET}_prune_{PRUNE}.csv')
    # output csv file named data/test.csv
    df_main.to_csv(f'data/hot_encoded_alcohol_dataset_{DATASET}_prune_{PRUNE}.csv', index=False)

    inFile = dataFromFile(f'data/alcohol_dataset_apriori_{DATASET}_prune_{PRUNE}.csv')
    items, rules = runApriori(inFile, MIN_SUPP, MIN_CONF)

    outputItemsRules(items, rules, MIN_SUPP, MIN_CONF, DATASET, PRUNE)
