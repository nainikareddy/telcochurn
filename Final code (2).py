# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 15:22:19 2018

@author: desai
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#data exploration

#Checking for missing values present in the data
print("No of missing values: ", dataset.isnull().sum().values.sum())
#Number of missing values present: 11 in TotalCharges column.

#Checking for number of rows, columns, features & if the features are unique
print("The number of Rows:", dataset.shape[0])
print("The number of Columns:", dataset.shape[1])
print("Features:" , dataset.columns.tolist())
print("Unique values:",dataset.nunique())

#*****************************************************************************
# Replacing "No phone/internet Service" value with No in the columns 
dataset['MultipleLines'].replace(['No phone service'], 'No',inplace=True)
dataset['OnlineSecurity'].replace(['No internet service'], 'No',inplace=True)
dataset['OnlineBackup'].replace(['No internet service'], 'No',inplace=True)
dataset['DeviceProtection'].replace(['No internet service'], 'No',inplace=True)
dataset['TechSupport'].replace(['No internet service'], 'No',inplace=True)
dataset['StreamingTV'].replace(['No internet service'], 'No',inplace=True)
dataset['StreamingMovies'].replace(['No internet service'], 'No',inplace=True)

#Replacing empty values in TotalCharges columns with Nan & dropping those rows
dataset['TotalCharges']=dataset['TotalCharges'].replace(" ",np.nan)
dataset = dataset[dataset["TotalCharges"].notnull()]
dataset= dataset.reset_index()[dataset.columns]

#Converting TotalCharges to float type
dataset['TotalCharges']=pd.to_numeric(dataset['TotalCharges'])

#Conveting binary values from SeniorCitizen column to yes or no
dataset['SeniorCitizen']= dataset['SeniorCitizen'].replace({1:"Yes",0:"No"})

#Manually renaming Male/Female values with 0 & 1 in gender column
dataset['gender']= dataset['gender'].replace({'Male':0, 'Female':1})

#Creating a bin for tenure column for data generalization & to avoid overfitting
def tenure_lab(dataset) : 
    if dataset["tenure"] <= 12 :
        return "Tenure_0-12"
    elif (dataset["tenure"] > 12) & (dataset["tenure"] <= 24 ):
        return "Tenure_12-24"
    elif (dataset["tenure"] > 24) & (dataset["tenure"] <= 48) :
        return "Tenure_24-48"
    elif (dataset["tenure"] > 48) & (dataset["tenure"] <= 60) :
        return "Tenure_48-60"
    elif dataset["tenure"] > 60 :
        return "Tenure_gt_60"
dataset["tenure_group"] = dataset.apply(lambda dataset:tenure_lab(dataset),axis = 1)

#Replacing tenure column with tenure_group
dataset["tenure"]=dataset["tenure_group"]
dataset=dataset.drop(['tenure_group'],axis = 1)

#Checking for correlation between the column labels
correlation = dataset.corr()
##Dropping of the 2 highly correlated variables
dataset=dataset.drop(['MonthlyCharges'],axis = 1)

#Assigning features and target variable
X = dataset.iloc[:, 1:19].values
y = dataset.iloc[:, 19].values

#Preprocessing with label encoder & onehotencoder
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])

from sklearn.preprocessing import LabelEncoder
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])

labelencoder_X_5 = LabelEncoder()
X[:, 5] = labelencoder_X_5.fit_transform(X[:, 5])

labelencoder_X_6 = LabelEncoder()
X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6])

labelencoder_X_7 = LabelEncoder()
X[:, 7] = labelencoder_X_7.fit_transform(X[:, 7])

labelencoder_X_8 = LabelEncoder()
X[:, 8] = labelencoder_X_8.fit_transform(X[:, 8])

labelencoder_X_9 = LabelEncoder()
X[:, 9] = labelencoder_X_9.fit_transform(X[:, 9])

labelencoder_X_10 = LabelEncoder()
X[:, 10] = labelencoder_X_10.fit_transform(X[:, 10])

labelencoder_X_11 = LabelEncoder()
X[:, 11] = labelencoder_X_11.fit_transform(X[:, 11])

labelencoder_X_12 = LabelEncoder()
X[:, 12] = labelencoder_X_12.fit_transform(X[:, 12])

labelencoder_X_13 = LabelEncoder()
X[:, 13] = labelencoder_X_13.fit_transform(X[:, 13])

labelencoder_X_14 = LabelEncoder()
X[:, 14] = labelencoder_X_14.fit_transform(X[:, 14])

labelencoder_X_15 = LabelEncoder()
X[:, 15] = labelencoder_X_15.fit_transform(X[:, 15])

labelencoder_X_16 = LabelEncoder()
X[:, 16] = labelencoder_X_16.fit_transform(X[:, 16])

#OneHotEncoder - Tenure
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [4])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#OneHotEncoder - InternetService
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [10])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#OneHotEncoder - Contract
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [18])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#OneHotEncoder - Payment
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [21])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#LabelEncoding Dependent Variable - Churn
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Standard Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#*******************************************************************************
#Training the model
#KNN

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_knn = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report
print("Training accuracy: ",classifier.score(X_train,y_train))
print("Testing Accuracy: ",accuracy_score(y_test, y_pred_knn))
print("\nClassification Report: ", classification_report(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

#Training accuracy:  0.8307555555555556
#Testing Accuracy:  0.7540867093105899
'''              precision    recall  f1-score   support

          0       0.82      0.85      0.84      1038
          1       0.53      0.49      0.51       369

avg / total       0.75      0.75      0.75      1407'''

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_knn, pos_label=1)
print("AUC :",metrics.auc(fpr, tpr))

#AUC : 0.6674029168037344

#*******************************************************************************
#Training the Model
#Decision Tree

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_decision = classifier.predict(X_test)

print("Training accuracy: ",classifier.score(X_train,y_train))
print("Testing Accuracy: ",accuracy_score(y_test, y_pred_decision))
print("\nClassification Report: ", classification_report(y_test, y_pred_decision))

#Training accuracy:  0.9978666666666667
#Testing Accuracy:  0.7412935323383084

''' precision    recall  f1-score   support

          0       0.83      0.82      0.82      1038
          1       0.51      0.51      0.51       369

avg / total       0.74      0.74      0.74      1407'''

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_decision = confusion_matrix(y_test, y_pred_decision)

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_decision, pos_label=1)
print("AUC :",metrics.auc(fpr, tpr))

#AUC : 0.6665922584081333

#******************************************************************************
#Training the model
#Random Forest

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_random = classifier.predict(X_test)

print("Training accuracy: ",classifier.score(X_train,y_train))
print("Testing Accuracy: ",accuracy_score(y_test, y_pred_decision))
print("\nClassification Report: ", classification_report(y_test, y_pred_random))

#Training accuracy:  0.9793777777777778
#Testing Accuracy:  0.7412935323383084

'''precision    recall  f1-score   support

          0       0.82      0.89      0.85      1038
          1       0.59      0.46      0.51       369

avg / total       0.76      0.77      0.76      1407'''

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_random = confusion_matrix(y_test, y_pred_random)

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_random, pos_label=1)
print("AUC :",metrics.auc(fpr, tpr))

#AUC : 0.6712838949198741

#******************************************************************************
#Training the model
#ANN

#Importing the Keras libraries and packages
import keras #Using Either Tensorflow & Theano
from keras.models import Sequential #Required to initialize NN
from keras.layers import Dense #Required to build the layers of ANN

# Initialising the ANN :Defining it as a sequence of layers
classifier = Sequential()

#Adding the input layer and the first hidden layer.
classifier.add(Dense(output_dim = 13, init = 'uniform', activation = 'relu', input_dim = 25))

classifier.add(Dense(output_dim = 13, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred_ANN = classifier.predict(X_test)
y_pred_ANN = (y_pred_ANN > 0.5)

print("Testing Accuracy: ",accuracy_score(y_test, y_pred_ANN))
#Testing Accuracy:  0.7974413646055437

print("\nClassification Report: ", classification_report(y_test, y_pred_ANN))

'''precision    recall  f1-score   support

          0       0.83      0.91      0.87      1038
          1       0.65      0.49      0.56       369

avg / total       0.79      0.80      0.79      1407'''

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_ANN = confusion_matrix(y_test, y_pred_ANN)

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_ANN, pos_label=1)
print("AUC :",metrics.auc(fpr, tpr))

#AUC : 0.6967863464761814

#*****************************************************************************
