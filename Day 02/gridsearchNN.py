# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:16:02 2020

@author: user
"""


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import svm

#import data using pandas
training_data = (pd.read_csv('credit_card_default_train.csv')) #load train csv file and convert it to a 2D matrix
test_data = (pd.read_csv('credit_card_default_test.csv')) #load test csv file and convert it to a 2D matrix
training_data.head()
test_data.head()

#divide dataset to labels and features and remove userid(unique value)
labels = training_data.iloc[:,-1].values
features =training_data.iloc[:,1:-1].values
x_test =test_data.iloc[:,1:].values

#imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

#create function for normalize data in dataset.
def normalize(x_train):
    for column in range(11,29):
        avg = sum(x_train[:,column])/x_train.shape[0]
        #print (avg)
        datarange = (max(x_train[:,column]) - min(x_train[:,column]))
        #print(datarange)
        for data in range(0,x_train.shape[0]):
            x_train[data,column] = (x_train[data,column] - avg)/datarange
    return x_train

features = normalize(features)
#x_test =normalize(x_test)

#encode string into numbers in dataset using label encoder.
encode= LabelEncoder()
features[:,0] = encode.fit_transform(features[:,0])  #encode balance limit in training set
features[:,1] = encode.fit_transform(features[:,1])  #encode gender in training set
features[:,2] = encode.fit_transform(features[:,2])  #encode education status in training set
features[:,3] = encode.fit_transform(features[:,3])  #encode maritial status in training set
features[:,4] = encode.fit_transform(features[:,4])  #encode age in training set

x_test[:,0] = encode.fit_transform(x_test[:,0])     #encode balance limit in test set
x_test[:,1] = encode.fit_transform(x_test[:,1])     #encode gender in test set
x_test[:,2] = encode.fit_transform(x_test[:,2])     #encode education status in test set
x_test[:,3] = encode.fit_transform(x_test[:,3])     #encode maritial status in test set
x_test[:,4] = encode.fit_transform(x_test[:,4])     #encode age in test set

#convert data in features and x_test into floats
x_test= x_test.astype(np.float)
features = features.astype(np.float)

#split training data set in to train and vaidate
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

#list the different values for parameters to get best parameters values
param_grid = {'C':[1,2,3,4,5,6,7,8,9,10],'max_iter':[75,100,125,200,250,300,500,1000],'random_state':[0,1,2,3,4,5,6,7,8,9]}

#model implementing
model= Sequential([Flatten(),Dense(16,activation='sigmoid',input_shape=features.shape),Dense(8,activation='sigmoid'),Dense(1,activation='sigmoid')])

#model compiling
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

#use gridsearch to identify best parameters for model
grid_search = GridSearchCV(estimator=svm.SVC(),param_grid=[param_grid])
model.fit(X_train, Y_train,validation_data=(X_test,Y_test), epochs=25, verbose = 1)
grid_search.fit(X_train,Y_train)
print("After grid search")
model.fit(X_train, Y_train,validation_data=(X_test,Y_test), epochs=25, verbose = 1)

