import pandas as pd
import numpy as np
import sys
from dataStructs.Node import Node
from dataStructs.Edge import Edge
import json
from time import time

def read_csv(file_name, restrictions_file_name):
    # read second and third line of file_name
    with open(file_name, 'r') as f:
        # ignore first line
        f.readline()
        # read second line
        second_line = f.readline()
        # read third line
        third_line = f.readline()

    # split second line into list of integers
    second_line_list = second_line.split(',')
    attribute_domains = [int(x) for x in second_line_list]
    # split third line into a list csv and get first element
    target_var = third_line.split(',')[0].replace('\n', '')
    # strip quotes from target_var
    target_var = target_var.replace('"', '')

    # read file_name into dataframe skipping the first and second lines
    df = pd.read_csv(file_name, skiprows=[1, 2])

    # get attributes as columns
    attributes = df.columns.values.tolist()

    # if attribute domain is -1, then remove it from attributes and attribute domains
    for idx, attribute in enumerate(attributes):
        if attribute_domains[idx] == -1:
            attributes.remove(attribute)
            attribute_domains.remove(attribute_domains[idx])

    # remove target variable from attributes and attribute domains
    target_domain = attribute_domains.pop(attributes.index(target_var))
    attributes.remove(target_var)

    # TODO: restrictions file

    # replace all '?' with null
    df = df.replace('?', np.nan)
    # remove rows where any column value is ?
    df = df.dropna()
    # reindex dataframe
    df = df.reset_index(drop=True)

    return df, attributes, attribute_domains, target_var, target_domain

# get Attribute entropy
def getAttributeEntropy(df, attribute, target_var, target_domain):
    # dataframe length
    df_length = len(df)
    # total entropy
    total_entropy = 0
    # get entropy for each attribute value
    attr_val_entropies = {}
    # get unique values of attribute
    attribute_values = df[attribute].unique()
    # for each attribute value get entropy
    for attr_val in attribute_values:
        val_entropy = 0
        # get subset of df where attribute value is equal to attr_val
        subset = df[df[attribute] == attr_val]
        # get length of subset
        subset_length = len(subset)
        # get value counts of target_var in subset
        subset_counts = subset[target_var].value_counts()
        # for item in subset_counts
        for item in subset_counts.items():
            # get probability of item
            prob = item[1] / subset_length
            # get entropy and account for 0 probability
            if prob != 0:
                val_entropy += -prob * np.log2(prob)
            else:
                val_entropy += 0
        # add entropy to attr_val_entropies
        attr_val_entropies[attr_val] = val_entropy
        total_entropy += val_entropy * (subset_length / df_length)

    return (total_entropy, attr_val_entropies)

def findBestSplitContinous(df, attribute, target_var, target_domain, parent_entropy):
    # get unique values of attribute
    attribute_alphas = df[attribute].unique()
    alpha_entropies = {}
    df_length = len(df)
    max_info_gain_alpha = (None, 0)
    # for each alpha
    for alpha in attribute_alphas:
        total_entropy_alpha = 0

        # split dataframe by alpha
        smaller = df[df[attribute] <= alpha]
        larger = df[df[attribute] > alpha]
        len_smaller = len(smaller)
        len_larger = len(larger)

        # get target_var value counts for smaller and larger
        smaller_counts = smaller[target_var].value_counts()
        larger_counts = larger[target_var].value_counts()

        smaller_entropy = 0
        larger_entropy = 0

        # #numpy way but it was slower
        # if len_smaller != 0:
        #     smaller_counts_arr = smaller_counts.to_numpy() / len_smaller
        #     smaller_counts_arr_log = np.log2(smaller_counts_arr)
        #     smaller_counts_arr_neg = np.negative(smaller_counts_arr)

        #     smaller_entropy = np.sum(smaller_counts_arr_log * smaller_counts_arr_neg)
        # if len_larger != 0:
        #     larger_counts_arr = larger_counts.to_numpy() / len_larger
        #     larger_counts_arr_log = np.log2(larger_counts_arr)
        #     larger_counts_arr_neg = np.negative(larger_counts_arr)

        #     larger_entropy = np.sum(larger_counts_arr_log * larger_counts_arr_neg)

        # for each item in smaller_counts
        for item in smaller_counts.items():
            prob = item[1] / len_smaller
            if prob != 0:
                smaller_entropy += -prob * np.log2(prob)
            else:
                smaller_entropy += 0

        # for each item in larger_counts
        for item in larger_counts.items():
            prob = item[1] / len_larger
            if prob != 0:
                larger_entropy += -prob * np.log2(prob)
            else:
                larger_entropy += 0

        # get total entropy
        total_entropy_alpha = (len_smaller / df_length) * smaller_entropy + (len_larger / df_length) * larger_entropy
        # get info gain
        info_gain_alpha = parent_entropy - total_entropy_alpha
        if info_gain_alpha > max_info_gain_alpha[1]  or max_info_gain_alpha[0] is None:
            max_info_gain_alpha = (alpha, info_gain_alpha)

        alpha_entropies[alpha] = (total_entropy_alpha, {'val': alpha, 'less': smaller_entropy, 'greater': larger_entropy})

    return max_info_gain_alpha[0], alpha_entropies[max_info_gain_alpha[0]]

# select split attribute
def selectSplitAttribute(df, attributes, attribute_domains, target_var, target_domain, parent_entropy, thres):
    attribute_entropies = {}
    max_info_gain_tuple = (None, 0, None)
    # get attribute entropies
    for idx, attribute in enumerate(attributes):
        # if domain of attribute if 0, then find best split of continuous attribute
        if attribute_domains[idx] == 0:
            # TODO: factor in continous attribute
            bestAlpha, alphaEntropies = findBestSplitContinous(df, attribute, target_var, target_domain, parent_entropy)
            attribute_entropies[attribute] = alphaEntropies
            attribute_info_gain = parent_entropy - alphaEntropies[0]
            if attribute_info_gain > max_info_gain_tuple[1] or max_info_gain_tuple[0] is None:
                max_info_gain_tuple = (attribute, attribute_info_gain, True)
        else:
            attribute_entropies[attribute] = getAttributeEntropy(df, attribute, target_var, target_domain)
            # get attribute info gains
            attribute_info_gain = parent_entropy - attribute_entropies[attribute][0]
            if attribute_info_gain > max_info_gain_tuple[1] or max_info_gain_tuple[0] is None:
                max_info_gain_tuple = (attribute, attribute_info_gain, False)

    # continuous flag
    continuous = False
    # get attribute with max info gain
    pot_split_attr = max_info_gain_tuple[0]
    # domain of pot_split_attr is 0
    if max_info_gain_tuple[2]:
        continuous = True

    # if info gain is less than threshold, return None
    if max_info_gain_tuple[1] < thres:
        return None, None, None
    
    return pot_split_attr, attribute_entropies[pot_split_attr], continuous


def c45(df, attributes, attribute_domains, target_var, target_domain, parent_entropy, thres):
    df_target_val_counts = df[target_var].value_counts()
    df_len = len(df)
    # if all values of target_var are the same, return leaf node
    if df[target_var].nunique() == 1:
        leafNode = Node(df[target_var].iloc[0], True)
        leafNode.p = 1.0
        return leafNode
    # else if attributes is empty return leaf node with most frequent value of target_var
    elif len(attributes) == 0:
        leafNode = Node(df_target_val_counts.idxmax(), True)
        # get probability of most frequent value of target_var
        leafNode.p = df_target_val_counts.max() / df_len
        return leafNode
    else:
        split_attribute, split_entropies, continuous = selectSplitAttribute(df, attributes, attribute_domains, target_var, target_domain, parent_entropy, thres)
        # if split_attribute is None, return leaf node with most frequent value of target_var
        if split_attribute is None:
            leafNode = Node(df_target_val_counts.idxmax(), True)
            # get probability of most frequent value of target_var

            # get max target_val count
            leafNode.p = df_target_val_counts.max() / df_len
            return leafNode
        else:
            # get domain of split attribute
            split_attribute_values = list(split_entropies[1].keys())
            # get keys of split_entropies[1], unique values of attribute that is detected
            split_attribute_domain = attribute_domains[attributes.index(split_attribute)]
            treeNode = Node(split_attribute, False)

            # - DEALING WITH GHOST PATHS -
            # if len(split_attribute_values) is less than split_attribute_domain, then create ghost edge to leaf node of most frequent value of target_var
            # this will be a default path for no match on classify
            if len(split_attribute_values) < split_attribute_domain:
                # get most frequent value of target_var
                most_frequent_target_var = df_target_val_counts.idxmax()
                ghostEdge = Edge(None, True, None)
                ghostLeafNode = Node(most_frequent_target_var, True)
                ghostLeafNode.p = df_target_val_counts.max() / df_len
                ghostEdge.endNode = ghostLeafNode
                treeNode.children.append(ghostEdge)
            # if continous
            if continuous:
                newEdgeLess = Edge(split_entropies[1]['val'], False, True)
                newEdgeLess.direction = 'le'
                newEdgeGreater = Edge(split_entropies[1]['val'], False, True)
                newEdgeGreater.direction = 'gt'

                less_entropy = split_entropies[1]['less']
                greater_entropy = split_entropies[1]['greater']

                df_split_less = df[df[split_attribute] <= split_entropies[1]['val']]
                df_split_greater = df[df[split_attribute] > split_entropies[1]['val']]

                newEdgeLess.endNode = c45(df_split_less, attributes, attribute_domains, target_var, target_domain, less_entropy, thres)
                newEdgeGreater.endNode = c45(df_split_greater, attributes, attribute_domains, target_var, target_domain, greater_entropy, thres)

                treeNode.children.append(newEdgeLess)
                treeNode.children.append(newEdgeGreater)
            else:
                # for each unique value of split_attribute
                for split_attribute_value in split_attribute_values:
                    newEdge = Edge(split_attribute_value, False, False)

                    split_attr_val_entropy = split_entropies[1][split_attribute_value]

                    # get dataframe of rows where split attribute is split_attribute_value
                    df_split_subset = df[df[split_attribute] == split_attribute_value]

                    # get new attributes and attribute domains
                    new_attributes = attributes.copy()
                    new_attribute_domains = attribute_domains.copy()

                    # remove split_attribute from attributes and attribute domains
                    new_attributes.remove(split_attribute)
                    new_attribute_domains.pop(attributes.index(split_attribute))

                    newEdge.endNode = c45(df_split_subset, new_attributes, new_attribute_domains, target_var, target_domain, split_attr_val_entropy, thres)

                    treeNode.children.append(newEdge)

    return treeNode

# convert tree representation to dictionary
def treeToDict(tree, dict):
    # if tree is leaf node
    if tree.leaf:
        if type(tree.var) is np.int64:
            tree.var = int(tree.var)
        dict['leaf'] = {'decision': tree.var, 'p': tree.p}
    else:
        # add split attribute to dictionary
        dict['node'] = {'var': tree.var, 'edges': []}
        for edge in tree.children:
            # add child to dictionary
            # check if edge.value is of type np.int64
            if type(edge.value) is np.int64:
                edge.value = int(edge.value)
            if edge.continuous:
                newEdge = {'value': edge.value, 'direction': edge.direction}
            else:
                newEdge = {'value': edge.value}
            treeToDict(edge.endNode, newEdge)
            dict['node']['edges'].append({'edge': newEdge})
    # return dictionary
    return dict

# entry point
if __name__ == "__main__":
    THRESHOLD = 0.25
    file_name = 'vectorOutputs/vector.csv'
    groundTruth = 'vectorOutputs/groundTruth.csv'
    restrictions_file_name = None

    df, attributes, attribute_domains, target_var, target_domain = read_csv(file_name, restrictions_file_name)

    # get probability of each unique value of target_var
    target_var_prob = df[target_var].value_counts(normalize=True)
    # calculate entropy of target_var
    target_var_entropy = -np.sum(target_var_prob * np.log2(target_var_prob))

    tree = c45(df, attributes, attribute_domains, target_var, target_domain, target_var_entropy, THRESHOLD)
    tree_dict = treeToDict(tree, {'filename': file_name})

    # print json dump to terminal with indent 4
    print(json.dumps(tree_dict, indent=4))
    # json dump to file /out/c45Trees/<filename>_<threshold>.json
    with open('question4_out/c45Output/c45Trees/' + file_name.split('/')[-1].split('.')[0] + '_' + str(THRESHOLD) + '.json', 'w') as outfile:
        json.dump(tree_dict, outfile, indent=4)
        outfile.close()
