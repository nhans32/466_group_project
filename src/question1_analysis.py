import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np


def getBasicRuleStats(rules_fp):
    rule_arr = []
    from_greater_than_1 = []
    to_greater_than_1 = []
    both_greater_than_1 = []
    both_less_than_2 = []
    with open(rules_fp, 'r') as f:
        rules = f.readlines()
        for rule in rules:
            new_rule = rule.split(' with confidence: ')

            items = new_rule[0]
            items = items.split(' ====> ')
            from_items = items[0].split(', ')
            to_items = items[1].split(', ')
            from_items = tuple([item.strip('()').strip(',').strip("'") for item in from_items])
            to_items = tuple([item.strip('()').strip(',').strip("'") for item in to_items])

            confidence = float(new_rule[1].strip())

            rule_f = {'from': from_items, 'to': to_items, 'confidence': confidence}
            if len(from_items) > 1:
                from_greater_than_1.append(rule_f)
            if len(to_items) > 1:
                to_greater_than_1.append(rule_f)
            if len(from_items) > 1 and len(to_items) > 1:
                both_greater_than_1.append(rule_f)
            if len(from_items) < 2 and len(to_items) < 2:
                both_less_than_2.append(rule_f)
            rule_arr.append(rule_f)
    return rule_arr, from_greater_than_1, to_greater_than_1, both_greater_than_1, both_less_than_2


def getBasicItemStats(items_fp):
    item_arr = []
    item_greater_than_1 = []
    item_less_than_2 = []
    with open(items_fp, 'r') as f:
        items = f.readlines()
        for item in items:
            item_set = item.split(' with support: ')
            supp = float(item_set[1].strip())
            item_set = item_set[0].split(', ')
            item_set = tuple([item.strip('()').strip(',').strip("'") for item in item_set])

            if len(item_set) > 1:
                item_greater_than_1.append({'set': item_set, 'support': supp})
            if len(item_set) < 2:
                item_less_than_2.append({'set': item_set, 'support': supp})
            item_arr.append({'set': item_set, 'support': supp})
    return item_arr, item_greater_than_1, item_less_than_2


def calcRuleStats(rule_arr, att_to_category, DATASET):
    # print(rule_arr)
    new_rules_general = {}
    for rule in rule_arr:
        # to prevent double counting of generalized associations
        cur_rule_track = {}
        for rule_from in rule['from']:
            rule_from = rule_from.split('_')[0]
            categorized_from = att_to_category[rule_from]
            for rule_to in rule['to']:
                rule_to = rule_to.split('_')[0]
                categorized_to = att_to_category[rule_to]

                if (categorized_from, categorized_to) not in new_rules_general and (categorized_from, categorized_to) not in cur_rule_track:
                    new_rules_general[(categorized_from, categorized_to)] = 1
                    cur_rule_track[(categorized_from, categorized_to)] = 1
                elif (categorized_from, categorized_to) not in cur_rule_track:
                    new_rules_general[(categorized_from, categorized_to)] += 1
                    cur_rule_track[(categorized_from, categorized_to)] = 1

    # sort new_rules_general
    new_rules_general = {item[0]:item[1] for item in sorted(new_rules_general.items(), key=lambda x: x[1], reverse=True)}

    # output new_rules_general to file called generalized_rules_{dataset}.txt
    with open(f'../analysis_outputs/generalized_rules_{DATASET}.txt', 'w') as f:
        for rule, count in new_rules_general.items():
            f.write(f'{rule[0]} ====> {rule[1]}: {count}\n')
        f.close()

    unique_categories = {} # like a set that maintains insertion order
    for rule_key in new_rules_general.keys():
        if rule_key[0] not in unique_categories:
            unique_categories[rule_key[0]] = 1
        if rule_key[1] not in unique_categories:
            unique_categories[rule_key[1]] = 1

    # initialize dataframe of zeros with rows as from categories and columns as to categories
    df_from = pd.DataFrame(0, index=unique_categories.keys(), columns=unique_categories.keys())

    for rule_key_val, val in new_rules_general.items():
        df_from.loc[rule_key_val[0], rule_key_val[1]] = val

    print(new_rules_general)

    # figsize=(10, 10)
    fig, ax = plt.subplots()
    data = df_from.to_numpy()
    data = np.ma.masked_where(data == 0, data)
    cmap = mpl.cm.get_cmap("Oranges").copy()
    cmap.set_bad(color='white')
    im = ax.imshow(data, cmap=cmap)
    ax.set_xticks(np.arange(len(unique_categories.keys())), labels=unique_categories.keys())
    ax.set_yticks(np.arange(len(unique_categories.keys())), labels=unique_categories.keys())

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(unique_categories.keys())):
        for j in range(len(unique_categories.keys())):
            text = ax.text(j, i, df_from.to_numpy()[i, j],
                        ha="center", va="center", color="black")

    ax.set_title(f'Distribution of Generalized Singlet Rules - Alcohol Dataset {DATASET}')
    fig.tight_layout()
    # set y label as from rule
    ax.set_ylabel('From Attribute Type')
    # set x label as to rule
    ax.set_xlabel('To Attribute Type')
    # make cells with 0 transparent

    plt.show()


def combinedSepAnalysis(MIN_SUPP, MIN_CONF, PRUNE, attr_to_cat):
    # iterate through all files in outputs rules
    # get rules that appear in seperate, combined datasets
    combined_fp = f'../outputs/apriori_rules_supp_{MIN_SUPP}_conf_{MIN_CONF}_combined_prune_{PRUNE}.txt'
    mat_fp = f'../outputs/apriori_rules_supp_{MIN_SUPP}_conf_{MIN_CONF}_mat_prune_{PRUNE}.txt'
    por_fp = f'../outputs/apriori_rules_supp_{MIN_SUPP}_conf_{MIN_CONF}_por_prune_{PRUNE}.txt'

    rule_arr_combined, from_greater_than_1_combined, to_greater_than_1_combined, both_greater_than_1_combined, both_less_than_2_combined = getBasicRuleStats(combined_fp)
    rule_arr_mat, from_greater_than_1_mat, to_greater_than_1_mat, both_greater_than_1_mat, both_less_than_2_mat = getBasicRuleStats(mat_fp)
    rule_arr_por, from_greater_than_1_por, to_greater_than_1_por, both_greater_than_1_por, both_less_than_2_por = getBasicRuleStats(por_fp)

    rule_dict_combined = {(rule['from'], rule['to']):0 for rule in rule_arr_combined}
    rule_dict_mat = {(rule['from'], rule['to']):0 for rule in rule_arr_mat}
    rule_dict_por = {(rule['from'], rule['to']):0 for rule in rule_arr_por}

    # get keys that are present in all three dictionaries
    overlap_rules = rule_dict_combined.keys() & rule_dict_mat.keys() & rule_dict_por.keys()
    overlap_rules_stats = [{'from':rule[0], 'to':rule[1], 'confidence': 'NA'} for rule in overlap_rules]
    calcRuleStats(overlap_rules_stats, attr_to_cat, 'Overlap')

    out_fp = f'../analysis_outputs/rule_overlap_{MIN_SUPP}_conf_{MIN_CONF}_prune_{PRUNE}.txt'
    with open(out_fp, 'w') as f:
        f.write(str(len(overlap_rules)) + '\n')
        for rule in overlap_rules:
            f.write(f'{rule[0]} => {rule[1]}\n')


if __name__ == '__main__':
    MIN_SUPP = 0.45
    MIN_CONF = 0.8
    DATASET = 'por'
    PRUNE = True
    ANALYZE_COMBINED = True

    items_fp = f'../outputs/apriori_items_supp_{MIN_SUPP}_{DATASET}_prune_{PRUNE}.txt'
    rules_fp = f'../outputs/apriori_rules_supp_{MIN_SUPP}_conf_{MIN_CONF}_{DATASET}_prune_{PRUNE}.txt'

    categories = {'demographics':0, 'financial/accessibility':0, 'education_history':0, 'familial':0, 'social/extra-cirricular':0,
                  'alcohol_consumption':0, 'health':0, 'reason_for_attending_this_school':0, 'class_type':0, 'future_education_pursuit':0,
                  'current_education':0, 'grade':0}

    attribute_categorization = {
        'sex': 'demographics',
        'age': 'demographics',
        'school': 'demographics',
        'address': 'demographics',
        'traveltime': 'financial/accessibility',
        'paid': 'financial/accessibility',
        'internet': 'financial/accessibility',
        'failures': 'education_history',
        'nursery': 'education_history',
        'famsize': 'familial',
        'Pstatus': 'familial',
        'Medu': 'familial',
        'Fedu': 'familial',
        'Mjob': 'familial',
        'Fjob': 'familial',
        'guardian': 'familial',
        'famrel': 'familial',
        'activities': 'social/extra-cirricular',
        'romantic': 'social/extra-cirricular',
        'freetime': 'social/extra-cirricular',
        'goout': 'social/extra-cirricular',
        'Dalc': 'alcohol_consumption',
        'Walc': 'alcohol_consumption',
        'health': 'health',
        'reason': 'reason_for_attending_this_school',
        'class': 'class_type',
        'higher': 'future_education_pursuit',
        'studytime': 'current_education',
        'schoolsup': 'current_education',
        'famsup': 'current_education',
        'absences': 'current_education',
        'G1': 'grade',
        'G2': 'grade',
        'G3': 'grade'
    }

    rule_arr, from_greater_than_1, to_greater_than_1, both_greater_than_1, both_less_than_2 = getBasicRuleStats(rules_fp)
    print(f'Rule Left >1: {len(from_greater_than_1)}')
    print(f'Rule Right >1: {len(to_greater_than_1)}')
    print(f'Rule Both >1: {len(both_greater_than_1)}')
    print(f'Rule Both <2: {len(both_less_than_2)}')
    print(f'Num. Rules: {len(rule_arr)}\n')

    item_arr, item_greater_than_1, item_less_than_2 = getBasicItemStats(items_fp)
    print(f'Item >1: {len(item_greater_than_1)}')
    print(f'Item <2: {len(item_less_than_2)}')
    print(f'Num. Items: {len(item_arr)}')

    if ANALYZE_COMBINED:
        combinedSepAnalysis(MIN_SUPP, MIN_CONF, PRUNE, attribute_categorization)

    calcRuleStats(rule_arr, attribute_categorization, DATASET)


