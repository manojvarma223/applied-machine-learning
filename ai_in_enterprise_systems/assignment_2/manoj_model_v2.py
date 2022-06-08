### Notebook Imports

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from keras.activations import relu, sigmoid

### Importing the data
breast_cancer_dataset = load_breast_cancer()

features = breast_cancer_dataset.data
feature_names = breast_cancer_dataset.feature_names
target = breast_cancer_dataset.target
target_names = breast_cancer_dataset.target_names

X = pd.DataFrame(features, columns=feature_names)
y = pd.Series(target)


### Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=1)

### Feature Scaling

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

### Neural Network Model

num_classes = 2
input_size = 30

model = Sequential([
    keras.Input(shape=input_size),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train_scaled, y_train, epochs=20)

training_score = model.evaluate(x_train_scaled, y_train)
testing_score = model.evaluate(x_test_scaled, y_test)

print('Training Accuracy: ',training_score[1]*100)
print('Testing Accuracy: ',testing_score[1]*100)
