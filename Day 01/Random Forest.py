# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 19:04:43 2020

@author: user
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

training_data = (pd.read_csv('credit_card_default_train.csv')) #load train csv file and convert it to a 2D matrix
test_data = (pd.read_csv('credit_card_default_test.csv')) #load test csv file and convert it to a 2D matrix
y_sub =(pd.read_csv('submission.csv'))

y_sub2 =(pd.read_csv('neural_network_submission1.csv'))
y_sub = y_sub.iloc[:,1:].values
y_sub2 = y_sub2.iloc[:,1:].values
training_data.head()
test_data.head()
labels = training_data.iloc[:,12:13].values
features =training_data.iloc[:,1:12].values
x_test =test_data.iloc[:,1:].values

#imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
print(features[0])
def normalize(x_train):
    for column in range(5,11):
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


X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.3)



RF = RandomForestClassifier()
RF.fit(X_train,Y_train)
prediction= RF.predict(X_test)
#print(accuracy_score(y_sub2, prediction))
print(accuracy_score(Y_test, prediction))



#pd.DataFrame({'Client_ID':test_data.Client_ID,'NEXT_MONTH_DEFAULT':prediction2}).set_index('Client_ID').to_csv("randForestSubmission5.csv")