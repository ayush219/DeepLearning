# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:53:33 2019

@author: Ayush
"""
#Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer

#Dataset
dataset= pd.read_csv('Churn_Modelling.csv')
X= dataset.iloc[:,3:-1].values
Y= dataset.iloc[:,-1].values

#Encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1= LabelEncoder()
X[:,1]= labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2= LabelEncoder()
X[:,2]= labelencoder_X_2.fit_transform(X[:,2])

#onehotencoder= OneHotEncoder(categorical_features=[1])
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X=ct.fit_transform(X)
X=X[:,1:]

#Splitting data
from sklearn. model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=0)

#Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)

#Deep Learning Libraries
from keras.models import Sequential
from keras.layers import Dense

#ANN
classifier=Sequential()

#First layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
#Second layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
#Output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

#Compiling ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

#Fitting ANN
classifier.fit(X_train,Y_train, batch_size=10, nb_epoch=100)

#Prediction
Y_pred=classifier.predict(X_test)
Y_pred=(Y_pred>0.5)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test, Y_pred)




