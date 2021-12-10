import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

from linreg import LinearRegressor

with open("../data/alcohol_dataset.pkl", "rb") as file:
    df = pickle.load(file)
print(df.shape)
X = df.drop(["G3"], axis=1)
y = df["G3"]
# curse of dimensionality: 1000 rows, 3400 columns
# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(interaction_only=True, include_bias=False)
# X = poly.fit_transform(X)
# print(X.shape)

# model = LinearRegression()
# rfe = RFE(estimator=model, n_features_to_select=1, step=1)
# rfe.fit(X, y)
# imp_feat = list(zip(rfe.ranking_, X.columns.values))
# imp_feat = sorted(imp_feat, key=lambda x: x[0])

# target columns: 'G1', 'G2', 'G3'

# best = 0
# feats_b = None
# for i in range(len(imp_feat)):
#     feats = [j[1] for j in imp_feat[:i]]
#     model = LinearRegression()
#     score = cross_val_score(model, X[feats], y, cv=10).mean()
#     if score > best:
#         print(score)
#         best = score
#         feats_b = feats
# print(len(feats_b))

# news = []
# didn't help
# for rank, feat in imp_feat[:5]:
#     X[feat + "_sq"] = X[feat] ** 2
#     news.append(feat + "_sq")

# didn't help
# for i in range(3):
#     for j in range(i + 1, 3):
#         X[imp_feat[i][1] + imp_feat[j][1]] = X[imp_feat[i][1]] * X[imp_feat[j][1]]
#         news.append(imp_feat[i][1] + imp_feat[j][1])

# print(cross_val_score(model, X[feats_b + news], y, cv=10).mean())
# print(cross_val_score(model, X, y, cv=10).mean())

vals = ["Midterm 1", "Midterm 2", "Final Exam"]
full_data = [0.17395148481791783, 0.6870213077507976, 0.7922838776289416]
square = [0.14026224874587634, 0.7104135570131166, 0.8025987210884743]
interaction = [0.1680287421073609, 0.7107653134280063, 0.8019112415503891]
square_int = [-0.1429374177892186, 0.7105857143608902, 0.8017439618763202]
selection = [0.18517228818308354, 0.7105986506192415, 0.8034217686110285]
# g1 ['school', 'failures', 'schoolsup', 'paid', 'higher', 'absences', 'class_type', 'Medu_4', 'Fedu_0', 'Fedu_1', 'Fedu_2', 'Fedu_3', 'Fedu_4', 'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home', 'reason_other', 'reason_reputation', 'guardian_father', 'guardian_mother', 'guardian_other', 'traveltime_1', 'traveltime_2', 'traveltime_3', 'traveltime_4', 'studytime_1', 'studytime_2', 'studytime_3', 'studytime_4', 'famrel_1', 'famrel_2', 'famrel_3', 'famrel_4', 'famrel_5', 'freetime_1', 'freetime_2', 'freetime_3', 'freetime_4', 'freetime_5', 'goout_1', 'goout_2', 'goout_3', 'goout_4', 'goout_5', 'Dalc_1', 'Dalc_2', 'Dalc_3', 'Dalc_4', 'Dalc_5', 'Walc_1', 'Walc_2', 'Walc_3', 'Walc_4', 'Walc_5', 'health_1', 'health_2', 'health_3', 'health_4', 'health_5']
# g2 ['failures', 'absences', 'G1', 'Fedu_3', 'Fedu_4', 'Fjob_services', 'traveltime_4', 'studytime_3', 'famrel_4', 'famrel_5', 'freetime_2', 'goout_2', 'Walc_1', 'Walc_2', 'Walc_3', 'Walc_4', 'Walc_5', 'health_1', 'health_2', 'health_3', 'health_4', 'health_5']
# g3 ['age', 'famsize', 'schoolsup', 'paid', 'higher', 'internet', 'absences', 'G1', 'G2', 'class_type', 'Mjob_teacher', 'traveltime_4', 'studytime_2', 'freetime_2', 'freetime_3', 'goout_5', 'Dalc_2', 'Walc_1', 'Walc_2', 'Walc_3', 'Walc_4', 'Walc_5', 'health_1', 'health_2', 'health_3', 'health_4', 'health_5']
pruning = [0.19497246191441558, 0.7408417844517314, 0.816128409165815]
outliers = [0.17997246191441558, 0.7815790066438313, 0.9032479351069218]
# num_feats = [76, 66, 71]
# prune_feats = [68, 19, 27]

import matplotlib.pyplot as plt
plt.plot(vals, full_data, label="Full Dataset")
plt.plot(vals, square, label="Square Variables")
plt.plot(vals, interaction, label="Interaction Variables")
plt.plot(vals, square_int, label="Square + Interaction Variables")
plt.plot(vals, selection, label="Selection Dataset")
plt.plot(vals, pruning, label="Pruned Dataset")
plt.plot(vals, outliers, label="Pruned Outliers Dataset")
plt.legend()
plt.xlabel("Target Variable")
plt.ylabel("R^2")
plt.title("Amount of Variance Explained by Different Variable Sets")
plt.show()


def prune(X, columns, y, model, best_score=None, seen=None):
    if seen is None:
        seen = set()

    if len(columns) == 0 or tuple(columns) in seen:
        return [], best_score

    seen.add(tuple(columns))

    best_vars = []
    for column in columns:
        new_columns = [c for c in columns if c != column]
        model_score = (cross_val_score(model, X[new_columns], y, cv=10)).mean()
        if best_score is None or model_score > best_score:
            best_score = model_score
            print(best_score)
            best_vars.append(new_columns)

    temp = []
    for vars in best_vars:
        var_list, best_score = prune(X, vars, y, model, best_score, seen)
        temp += var_list

    if len(temp) == 0:
        return best_vars, best_score

    return temp, best_score


# Create the dataframe
column_name = X.columns
X = X.reset_index()[column_name]

''' Detection '''
# IQR
Q1 = np.percentile(y, 25,
                   interpolation='midpoint')

Q3 = np.percentile(y, 75,
                   interpolation='midpoint')
IQR = Q3 - Q1

print("Old Shape: ", X.shape)

# Upper bound
upper = np.where(y >= (Q3 + 1.5 * IQR))
# Lower bound
lower = np.where(y <= (Q1 - 1.5 * IQR))

''' Removing the Outliers '''
y = y.reset_index()
X.drop(list(upper[0]) + list(lower[0]), axis=0, inplace=True)
y.drop(list(upper[0]) + list(lower[0]), inplace=True)
y = y["G3"]

print(X.shape)

print(prune(X, X.columns.tolist(), y, LinearRegression()))

