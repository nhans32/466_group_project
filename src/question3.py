import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt


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
            # print(best_score)
            best_vars.append(new_columns)

    temp = []
    for vars in best_vars:
        var_list, best_score = prune(X, vars, y, model, best_score, seen)
        temp += var_list

    if len(temp) == 0:
        return best_vars, best_score

    return temp, best_score


if __name__ == "__main__":
    # load in the data
    with open("../data/alcohol_dataset.pkl", "rb") as file:
        df = pickle.load(file)

    # target columns: 'G1', 'G2', 'G3'
    X = df.drop(["G1", "G2", "G3"], axis=1)
    y = df["G1"]

    # find out how well the model fits to the whole dataset
    model = Ridge()
    print("Full dataset R^2:", cross_val_score(model, X, y, cv=10).mean())

    # # find all interaction terms with the dataset
    # # curse of dimensionality: 1000 rows, 3400 columns
    # from sklearn.preprocessing import PolynomialFeatures
    # poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    # X = poly.fit_transform(X)
    # print(X.shape)

    # rank features
    rfe = RFE(estimator=model, n_features_to_select=1, step=1)
    rfe.fit(X, y)
    imp_feat = list(zip(rfe.ranking_, X.columns.values))
    imp_feat = sorted(imp_feat, key=lambda x: x[0])

    # find the best features iteratively
    best = 0
    feats_b = None
    for i in range(1, len(imp_feat)):
        feats = [j[1] for j in imp_feat[:i]]
        score = cross_val_score(model, X[feats], y, cv=10).mean()
        if score > best:
            best = score
            feats_b = feats

    # add combinations of features based on their ranking
    new_feats = []
    # for rank, feat in imp_feat[:5]:
    #     X[feat + "_sq"] = X[feat] ** 2
    #     new_feats.append(feat + "_sq")

    for i in range(3):
        for j in range(i + 1, 3):
            X[imp_feat[i][1] + imp_feat[j][1]] = X[imp_feat[i][1]] * X[imp_feat[j][1]]
            new_feats.append(imp_feat[i][1] + imp_feat[j][1])

    print("Full dataset + square R^2:", cross_val_score(model, X[feats_b + new_feats].values, y, cv=10).mean())

    vals = ["Midterm 1", "Midterm 2", "Final Exam"]
    full_data = [0.17395148481791783, 0.6870213077507976, 0.7922838776289416]
    square = [0.14026224874587634, 0.7104135570131166, 0.8025987210884743]
    interaction = [0.1680287421073609, 0.7107653134280063, 0.8019112415503891]
    square_int = [-0.1429374177892186, 0.7105857143608902, 0.8017439618763202]
    selection = [0.18517228818308354, 0.7105986506192415, 0.8034217686110285]
    pruning = [0.19497246191441558, 0.7408417844517314, 0.816128409165815]
    outliers = [0.17997246191441558, 0.7815790066438313, 0.9032479351069218]

    # plot how the top 3 models did compared with one another
    plt.plot(vals, selection, label="Selection Dataset")
    plt.plot(vals, pruning, label="Pruned Dataset")
    plt.plot(vals, outliers, label="Pruned Outliers Dataset")
    plt.legend()
    plt.xlabel("Score")
    plt.ylabel("R^2")
    plt.title("R^2 for Each Target by Different Linear Regression Models")
    plt.show()

    # plot how well each model did on average
    plt.bar(
        ["Full Data", "Square", "Interaction", "Square and Interaction", "Selection", "Pruning", "Pruning w/ Outliers"],
        [np.mean([full_data]), np.mean([square]), np.mean([interaction]), np.mean([square_int]), np.mean([selection]), np.mean([pruning]), np.mean([outliers])],
        color="white",
        edgecolor="blue"
    )
    plt.xticks(rotation=10)
    plt.xlabel("Model Type")
    plt.ylabel("Mean R^2 of G1, G2 and G3")
    plt.title("Average Amount of Variance Explained by Different Linear Regression Models")
    plt.show()

    # # Removing outliers
    # column_name = X.columns
    # X = X.reset_index()[column_name]
    #
    # Q1 = np.percentile(y, 25, interpolation='midpoint')
    # Q3 = np.percentile(y, 75, interpolation='midpoint')
    # IQR = Q3 - Q1
    #
    # upper = np.where(y >= (Q3 + 1.5 * IQR))
    # lower = np.where(y <= (Q1 - 1.5 * IQR))
    #
    # y = y.reset_index()
    # X.drop(list(upper[0]) + list(lower[0]), axis=0, inplace=True)
    # y.drop(list(upper[0]) + list(lower[0]), inplace=True)
    # y = y["G3"]
    #
    # # prune based on removed outliers and grid search alpha values
    # for i in range(11):
    #     print(prune(X, X.columns.tolist(), y, Ridge(alpha=i)))
