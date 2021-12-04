import pickle
from sklearn.model_selection import cross_val_score

from linreg import LinearRegressor

with open("../data/alcohol_dataset.pkl", "rb") as file:
    df = pickle.load(file)

# target columns: 'G1', 'G2', 'G3'

# df = df.drop(["G2", "G3", "class_type"], axis=1)
model = LinearRegressor(iterations=2000, learning_rate=0.0001, lambda_coefficient=.3)
print(cross_val_score(model, df.drop(["G1", "G2", "G3"], axis=1), df["G1"], cv=10).mean())
