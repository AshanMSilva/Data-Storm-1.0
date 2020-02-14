# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 06:07:34 2020

@author: user
"""


import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


training_data = np.array(pd.read_csv('credit_card_default_train.csv')) #load train csv file and convert it to a 2D matrix
test_data = np.array(pd.read_csv('credit_card_default_test.csv')) #load test csv file and convert it to a 2D matrix

y_train=training_data[:,-1] #get the y values from training_data

#print(y_train)

drop_cols=[0,24] #colomns that should remove from x
x_train =np.delete(training_data,drop_cols,axis=1) #get x_train matrix

x_test = np.delete(test_data,[0],axis=1) #get x_test matrix

x_train_length =x_train.shape
y_train_length =y_train.shape
x_test_length =x_test.shape

#print(x_test_length);
#print(x_train_length);
#print(y_train_length);

i=0

for i in range(x_train_length[0]):
    if(type(x_train[i][0])==type("a")):
        if(x_train[i][0][-1]=='M'):
             x_train[i][0]= float(x_train[i][0][:-1])*10**6
        elif(x_train[i][0][-1]=='K' or x_train[i][0][-1]=='k'):
             x_train[i][0]= float(x_train[i][0][:-1])*10**3
    if(x_train[i][1]=="M"):
        x_train[i][1]= 1
    if(x_train[i][1]=="F"):
        x_train[i][1]= 0
    if(x_train[i][2]=="Other"):
        x_train[i][2]= 0
    if(x_train[i][2]=="Graduate"):
        x_train[i][2]= 1
    if(x_train[i][2]=="High School"):
        x_train[i][2]= 2
    if(x_train[i][3]=="Other"):
        x_train[i][3]= 0
    if(x_train[i][3]=="Single"):
        x_train[i][3]= 1
    if(x_train[i][3]=="Single"):
        x_train[i][3]= 1
    if(x_train[i][4]=="More than 65"): #479
        x_train[i][4]= 3
    if(x_train[i][4]=="46-65"):
        x_train[i][4]= 2
    if(x_train[i][4]=="31-45"):
        x_train[i][4]= 1
    if(x_train[i][4]=="Less than 30"):
        x_train[i][4]= 0  


x_train = x_train.astype(np.float)
#x_test = x_test.astype(np.float)
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=42)


X_train_length =X_train.shape


model= Sequential([Flatten(),Dense(16,activation='sigmoid',input_shape=X_train_length),Dense(16,activation='sigmoid'),Dense(1,activation='sigmoid')])
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
model.fit(X_train,Y_train,validation_data=(X_test,Y_test), epochs=10, verbose = 1)