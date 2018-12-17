import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13 ].values
y = dataset.iloc[:, 13].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()        


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifiers = Sequential()
    classifiers.add(Dense(units=11, kernel_initializer='uniform', activation='relu'))
    classifiers.add(Dropout(rate=0.2))
    classifiers.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifiers.add(Dropout(rate=0.2))
    classifiers.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifiers.add(Dropout(rate=0.2))
    classifiers.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifiers.add(Dropout(rate=0.2))
    classifiers.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifiers

classifiers = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [15, 32], 'epochs': [200,300], 'optimizer': ['adam', 'rmsprop', 'sgd'] }
gridSearch = GridSearchCV(estimator = classifiers, param_grid=parameters, scoring='accuracy', n_jobs=-1, cv=10)
gridSearch = gridSearch.fit(X_train, y_train)
params = gridSearch.best_params_
estimate = gridSearch.best_estimator_
index = gridSearch.best_index_
score = gridSearch.best_score_




    



 
