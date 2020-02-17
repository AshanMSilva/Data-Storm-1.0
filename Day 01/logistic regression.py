# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:16:02 2020

@author: user
"""


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression

training_data = (pd.read_csv('credit_card_default_train.csv')) #load train csv file and convert it to a 2D matrix
test_data = (pd.read_csv('credit_card_default_test.csv')) #load test csv file and convert it to a 2D matrix
training_data.head()
test_data.head()
labels = training_data.iloc[:,-1].values
features =training_data.iloc[:,1:-1].values
x_test =test_data.iloc[:,1:].values

#imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

def normalize(x_train):
    for column in range(11,23):
        avg = sum(x_train[:,column])/x_train.shape[0]
        datarange = (max(x_train[:,column]) - min(x_train[:,column]))
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

#X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

logReg =LogisticRegression()
logReg.fit(features, labels)

prediction= logReg.predict(x_test)


#print(test_data.Client_ID)
pd.DataFrame({'Client_ID':test_data.Client_ID,'NEXT_MONTH_DEFAULT':prediction}).set_index('Client_ID').to_csv("submission.csv")


