import sklearn as sk
import pandas as pd
from sklearn import preprocessing

if __name__ == '__main__':
    le = preprocessing.LabelEncoder()
    # Converting string labels into numbers.
    df = pd.read_csv("../data/student-mat.csv", skiprows=[1, 2])
    weather_encoded=le.fit_transform(df)
    print(weather_encoded)

    # converting string labels into numbers
    temp_encoded=le.fit_transform(temp)
    label=le.fit_transform(play)



    #combinig weather and temp into single listof tuples
    features=list(zip(weather_encoded,temp_encoded))

    model = sk.KNeighborsClassifier(n_neighbors=3)

    # Train the model using the training sets
    model.fit(features,label)

    #Predict Output
    predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
    print(predicted)
