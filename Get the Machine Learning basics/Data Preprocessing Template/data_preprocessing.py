#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:18:54 2020

@author: nishita.dutta
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas  as pd

#Converting the dataset into arrays
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
X[:, 0] = LabelEncoder().fit_transform(X[:, 0])

#OneHOtENcoder needed for categorical input values as the cateogries have no correlation with one another and 
#it would be wrong to encode them into 0,1,2 or such. SO we encode them into 001,010,100
onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])],remainder='passthrough')
X = onehotencoder.fit_transform(X)

#Predicted value can be simply label encoded into zeroes and ones 
y = LabelEncoder().fit_transform(y)


#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


#Feature scaling means bring all the features into the same range or scale (between -1 and +1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



