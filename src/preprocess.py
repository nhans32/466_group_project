import pandas as pd

df_1 = pd.read_csv("../data/student-mat.csv")
df_2 = pd.read_csv("../data/student-por.csv")

# binary flag for class
df_1["class_type"] = 0
df_2["class_type"] = 1

df = pd.concat([df_1, df_2])

# ================ Binary Classes ================
school_flags = {"GP": 0, "MS": 1}
df["school"] = df["school"].apply(lambda x: school_flags[x])

sex_flags = {"F": 0, "M": 1}
df["sex"] = df["sex"].apply(lambda x: sex_flags[x])

address_flags = {"U": 0, "R": 1}  # U for urban, R for rural
df["address"] = df["address"].apply(lambda x: address_flags[x])

family_flags = {"LE3": 0, "GT3": 1}  # LE3 for family size <= 3, GT3 for > 3
df["famsize"] = df["famsize"].apply(lambda x: family_flags[x])

parent_status_flags = {"T": 0, "A": 1}  # T for living together, A for apart
df["Pstatus"] = df["Pstatus"].apply(lambda x: parent_status_flags[x])

school_support_flags = {"no": 0, "yes": 1}  # extra educational support
df["schoolsup"] = df["schoolsup"].apply(lambda x: school_support_flags[x])

family_support_flags = {"no": 0, "yes": 1}  # family educational support
df["famsup"] = df["famsup"].apply(lambda x: family_support_flags[x])

paid_flags = {"no": 0, "yes": 1}  # extra paid classes for the course
df["paid"] = df["paid"].apply(lambda x: paid_flags[x])

activity_flags = {"no": 0, "yes": 1}  # extra-curricular activities
df["activities"] = df["activities"].apply(lambda x: activity_flags[x])

nursery_flags = {"no": 0, "yes": 1}  # attended nursery school
df["nursery"] = df["nursery"].apply(lambda x: nursery_flags[x])

higher_flags = {"no": 0, "yes": 1}  # wants higher education
df["higher"] = df["higher"].apply(lambda x: higher_flags[x])

internet_flags = {"no": 0, "yes": 1}  # has internet access at home
df["internet"] = df["internet"].apply(lambda x: internet_flags[x])

romantic_relationship_flags = {"no": 0, "yes": 1}
df["romantic"] = df["romantic"].apply(lambda x: romantic_relationship_flags[x])

def change_to_indicator(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """One-hot encoding for a categorical, non-binary column in the dataframe.

    Args:
        df:     dataframe to adjust
        col:    column to make into indicators

    Returns:
        pd.DataFrame: pivoted table
    """
    pivoted = pd.get_dummies(df[col])
    pivoted.rename(
        {column: col + "_" + str(column) for column in pivoted.columns}, axis=1, inplace=True
    )
    return pd.concat([df.drop([col], axis=1), pivoted], axis=1)

multi_category_columns = [
    "Medu", "Fedu", "Mjob", "Fjob", "reason",
    "guardian", "traveltime", "studytime", "famrel", "freetime",
    "goout", "Dalc", "Walc", "health"
]
# pivoting into binary classifiers
for multi_category_column in multi_category_columns:
    df = change_to_indicator(df, multi_category_column)

import pickle
with open("alcohol_dataset.pkl", "wb") as file:
    pickle.dump(df, file)
