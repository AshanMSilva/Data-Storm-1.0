import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.preprocessing import LabelEncoder


training_data = (pd.read_csv('/kaggle/input/data-storm-10/credit_card_default_train.csv')) #load train csv file and convert it to a 2D matrix
test_data = (pd.read_csv('/kaggle/input/data-storm-10/credit_card_default_test.csv')) #load test csv file and convert it to a 2D matrix
training_data.head()
test_data.head()
labels = training_data.iloc[:,-1].values
features =training_data.iloc[:,1:-1].values
x_test =test_data.iloc[:,1:].values

#imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

def normalize(x_train):                     #function to normalize features
    for column in range(11,23):
        avg = sum(x_train[:,column])/x_train.shape[0]
        datarange = (max(x_train[:,column]) - min(x_train[:,column]))
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

#convert test data set into float data type
x_test= x_test.astype(np.float)
#convert training data set into float data type
features = features.astype(np.float)

#split training data to check accuracy into validatin set
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.3, random_state=1)

#implement model
model= Sequential()
model.add(Flatten())
          
model.add(Dense(32,input_shape=X_train.shape))
model.add(Activation("relu"))
model.add(Dropout(0.4))

model.add(Dense(64))
model.add(Activation("relu")) 
model.add(Dropout(0.4))

model.add(Dense(16))
model.add(Activation("relu"))
model.add(Dropout(0.4))
           
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

#fit training data to the implemented model
model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=100, verbose = 1)

#predict output for test data set
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
df = pd.DataFrame({'Client_ID':test_data.Client_ID,'NEXT_MONTH_DEFAULT':next_default}).set_index('Client_ID').to_csv('/kaggle/working/Sheet17.csv')
