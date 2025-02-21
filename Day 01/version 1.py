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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
 

training_data = (pd.read_csv('credit_card_default_train.csv')) #load train csv file and convert it to a 2D matrix
test_data = (pd.read_csv('credit_card_default_test.csv')) #load test csv file and convert it to a 2D matrix
training_data.head()
test_data.head()
labels = training_data.iloc[:,-1].values
features =training_data.iloc[:,1:-1].values
x_test =test_data.iloc[:,1:].values

#imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

def normalize(x_train):                 #function to normalize features
    for column in range(11,23):
        avg = sum(x_train[:,column])/x_train.shape[0]
        #print (avg)
        datarange = (max(x_train[:,column]) - min(x_train[:,column]))
        #print(datarange)
        for data in range(0,x_train.shape[0]):
            x_train[data,column] = (x_train[data,column] - avg)/datarange
    return x_train

features = normalize(features)          #normalize training data
x_test =normalize(x_test)               #normalize test data


#encode string into numbers in dataset using label encoder.
encode= LabelEncoder()
features[:,0] = encode.fit_transform(features[:,0])         #encode balance limit in training set
features[:,1] = encode.fit_transform(features[:,1])         #encode gender in training set
features[:,2] = encode.fit_transform(features[:,2])         #encode education status in training set
features[:,3] = encode.fit_transform(features[:,3])         #encode maritial status in training set
features[:,4] = encode.fit_transform(features[:,4])         #encode age in training set

x_test[:,0] = encode.fit_transform(x_test[:,0])             #encode balance limit in test set
x_test[:,1] = encode.fit_transform(x_test[:,1])             #encode gender in test set
x_test[:,2] = encode.fit_transform(x_test[:,2])             #encode education status in test set
x_test[:,3] = encode.fit_transform(x_test[:,3])             #encode maritial status in test set
x_test[:,4] = encode.fit_transform(x_test[:,4])             #encode age in test set
x_test= x_test.astype(np.float)

#convert test data set into float data type
features = features.astype(np.float)

#split training data to check accuracy into validatin set
#X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

#implement model
model= Sequential([Flatten(),Dense(16,activation='sigmoid',input_shape=features.shape),Dense(16,activation='sigmoid'),Dense(1,activation='sigmoid')])
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
#fit training data to the model
model.fit(features, labels, epochs=10, verbose = 1)
#predict output for test data
prediction = model.predict(x_test)

#convert output np array into a list to save in a csv file
results =np.zeros(prediction.shape)
j=0
next_default=[]
for j in range(prediction.shape[0]):
    if(prediction[j]<0.5):
        results[j]=0
    else:
        results[j] =1



for i in range(len(results)):
    if (results[i][0]==0.0):
        next_default.append(0)
    else:
        next_default.append(1)

#create data frame and save output in a csv file            
df = pd.DataFrame({'Client_ID':test_data.Client_ID,'NEXT_MONTH_DEFAULT':next_default}).set_index('Client_ID').to_csv('Sheet2.csv')

