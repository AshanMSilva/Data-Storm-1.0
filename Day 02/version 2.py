# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:16:02 2020

@author: user
"""


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression

training_data = (pd.read_csv('credit_card_default_train.csv')) #load train csv file and convert it to a 2D matrix
test_data = (pd.read_csv('credit_card_default_test.csv')) #load test csv file and convert it to a 2D matrix
y_sub =(pd.read_csv('submission.csv'))

y_sub2 =(pd.read_csv('neural_network_submission1.csv'))
y_sub = y_sub.iloc[:,1:].values
y_sub2 = y_sub2.iloc[:,1:].values
training_data.head()
test_data.head()
labels = training_data.iloc[:,-1].values
features =training_data.iloc[:,1:-1].values
x_test =test_data.iloc[:,1:].values

#imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

def normalize(x_train):
    for column in range(11,68):
        avg = sum(x_train[:,column])/x_train.shape[0]
        #print (avg)
        datarange = (max(x_train[:,column]) - min(x_train[:,column]))
        #print(datarange)
        for data in range(0,x_train.shape[0]):
            x_train[data,column] = (x_train[data,column] - avg)/datarange
    return x_train

features = normalize(features)
x_test =normalize(x_test)

encode= LabelEncoder()
features[:,0] = encode.fit_transform(features[:,0])
features[:,1] = encode.fit_transform(features[:,1])
features[:,2] = encode.fit_transform(features[:,2])
features[:,3] = encode.fit_transform(features[:,3])
features[:,4] = encode.fit_transform(features[:,4])

x_test[:,0] = encode.fit_transform(x_test[:,0])
x_test[:,1] = encode.fit_transform(x_test[:,1])
x_test[:,2] = encode.fit_transform(x_test[:,2])
x_test[:,3] = encode.fit_transform(x_test[:,3])
x_test[:,4] = encode.fit_transform(x_test[:,4])

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.3, random_state=0)



#param_grid ={'C':[1,2,3,4,5,6,7,8,9,10],"max_iter":[75,100,125,150,200,300,500,1000],"random_state":[0,1,2,3,4,5,6,7,8,9,10],"tol":[0.0001,0.001,0.01,0.1,1,2,3,5,10]}


logReg =LogisticRegression(C=9, max_iter=125, tol=0.001,random_state=0)

#grid_search =GridSearchCV(estimator=logReg, param_grid=param_grid)


logReg.fit(X_train, Y_train)
#grid_search.fit(X_train,Y_train)
#print(grid_search.best_params_)

prediction= logReg.predict(X_test)
prediction2= logReg.predict(X_train)
#print(prediction)
print(accuracy_score(Y_test, prediction))
print(accuracy_score(Y_train, prediction2))



prediction3= logReg.predict(x_test)
print(accuracy_score(y_sub2, prediction3))
print(accuracy_score(y_sub, prediction3))

pd.DataFrame({'Client_ID':test_data.Client_ID,'NEXT_MONTH_DEFAULT':prediction3}).set_index('Client_ID').to_csv("logregSubmission11.csv")
