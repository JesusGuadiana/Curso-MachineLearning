# Data Preprocessing

#Importing the libaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset

dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean",axis = 0)
imputer = imputer.fit(x[:, 1:3])

x[:, 1:3] = imputer.transform(x[:, 1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_country = LabelEncoder()

x[: ,0] = labelencoder_country.fit_transform(x[:, 0])
onehotenconder = OneHotEncoder(categorical_features = [0])
x = onehotenconder.fit_transform(x).toarray()

labelencoder_purchase = LabelEncoder()
y = labelencoder_purchase.fit_transform(y)

#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_text, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
























