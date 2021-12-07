import matplotlib.pyplot as plt

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

def calcRuleStats(rule_arr, from_greater_than_1, to_greater_than_1, both_greater_than_1, both_less_than_2, att_to_category):
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

                if f'{categorized_from} -> {categorized_to}' not in new_rules_general and f'{categorized_from} -> {categorized_to}' not in cur_rule_track:
                    new_rules_general[f'{categorized_from} -> {categorized_to}'] = 1
                    cur_rule_track[f'{categorized_from} -> {categorized_to}'] = 1
                elif f'{categorized_from} -> {categorized_to}' not in cur_rule_track:
                    new_rules_general[f'{categorized_from} -> {categorized_to}'] += 1
                    cur_rule_track[f'{categorized_from} -> {categorized_to}'] = 1
    # sort new_rules_general
    new_rules_general = {item[0]:item[1] for item in sorted(new_rules_general.items(), key=lambda x: x[1], reverse=True)}
    print(new_rules_general)
    # # dictionaries maintain order in python 3.9, so can do this
    # rule_keys = list(new_rules_general.keys())

    # rule_vals = list(new_rules_general.values())
    # rule_vals_sum = sum(rule_vals)
    # for val in 

    # # pruning off rules for display less than 1.5% of total

    # print(rule_keys)
    # print(rule_vals)

    # fig1, ax1 = plt.subplots()
    # ax1.set_title('Distribution of Generalized Singlet Rules - Alcohol Dataset Combined')
    # ax1.pie(rule_vals, labels=rule_keys, autopct='%1.1f%%',
    #     shadow=True, startangle=90)
    # ax1.axis('equal')
    # plt.show()

    


    

if __name__ == '__main__':
    items_fp = 'outputs/apriori_items_supp_0.45.txt'
    rules_fp = 'outputs/apriori_rules_supp_0.45_conf_0.85.txt'

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

    calcRuleStats(rule_arr, from_greater_than_1, to_greater_than_1, both_greater_than_1, both_less_than_2, attribute_categorization)


