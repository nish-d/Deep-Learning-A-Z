# -*- coding: utf-8 -*-

"""
Created on Tue May  3 11:48:12 2020

@author: nishita.dutta

Section 4: Building an ANN 
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

    
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1])],remainder='passthrough')
X = onehotencoder.fit_transform(X)

#Solving for dummy variable trap
'''Suppose your column breaks into n columns. Now if you know n-1 values, you can tell the value 
of the nth column. This makes that column redundant and also introduces some form of correlation if you think 
about it. It is a basic requirement that all features must be independent of each other. 
So you choose to drop any one of the n columns. Here we choose to drop the first column.
'''
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing keras libraries and modules
import keras
from keras.models import Sequential
from keras.layers import Dense

# Fitting classifier to the Training set
# Create your classifier here

#Initilising the ANN

classifier = Sequential()

#Adding the input layer and the first hidden layer

#No. of nodes in the hidden layer is average of no. of nodes in input layer(11) and output layer(6)
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

#Adding second hidden layer. We dont need to mention input dim parameter as it is the second layer and the 
#classifier already knows the input dim on basis of the first hidden layer that ew creaed above.
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

#Final layer
#If dependent variable i.e. y has more than 2 classes, use softmax as activation function and units = no. of classes of the dependent variable
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

"""Compiling the ANN. 

Applying stochastic gradient descent on the ANN and adjusting the weights.

If dependent variable has 2 outcomes then logarithmic loss fun is called binary_crossentropy
If it has more than 2, then it called categorical_crossentropy

Optimizer is the function with which we adjust the weights. Here we use a type of stochastic gradient descent called adam.
Loss function applied is logarithmic loss function

For more information on optimsers : https://keras.io/optimizers/"""

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Batch size is the number of observations/rows after which we want to update the weights.
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Converting the probabilities to true or false
y_pred = (y_pred > 0.5)
y_test = (y_test == 1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



#HOMEWORK

"""
Use our ANN model to predict if the customer with the following informations will leave the bank: 

Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
So should we say goodbye to that customer ?
"""

dataset_hw = pd.read_csv('hw_dataset.csv')
hw = dataset_hw.iloc[:, :].values
hw[:,1] = labelencoder_X_1.transform(hw[:,1])
hw[:,2] = labelencoder_X_2.transform(hw[:,2])
hw = onehotencoder.transform(hw)

hw = sc.transform(hw)
hw = hw[:, 1:]

hw_pred = classifier.predict(hw)

