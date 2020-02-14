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
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

training_data = (pd.read_csv('credit_card_default_train.csv')) #load train csv file and convert it to a 2D matrix
test_data = (pd.read_csv('credit_card_default_test.csv')) #load test csv file and convert it to a 2D matrix
training_data.head()
test_data.head()
labels = training_data.iloc[:,-1].values
features =training_data.iloc[:,1:-1].values
x_test =test_data.iloc[:,1:].values

#imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)



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
#print(features[:,2])
x_test= x_test.astype(np.float)
#print(x_train.iloc[0,:]['Gender'])
#print(features)
print(x_test.shape)
print(features.shape)
features = features.astype(np.float)
#x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1)

model= Sequential([Flatten(),Dense(16,activation='sigmoid',input_shape=features.shape),Dense(16,activation='sigmoid'),Dense(1,activation='sigmoid')])
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
model.fit(features, labels, epochs=2, verbose = 1)
prediction = model.predict(x_test)
print(model.predict(x_test))

results =np.zeros(prediction.shape)
j=0
for j in range(prediction.shape[0]):
    if(prediction[j]<0.5):
        results[j]=0
    else:
        results[j] =1

print(results)
