### Imports

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sklearn.metrics import classification_report
from sklearn import metrics

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

# ### Build the model : XGBoost
xgb = XGBClassifier(min_child_weight=1, max_depth=12, learning_rate=0.2, gamma=0.0, colsample_bytree=0.3)

xgb.fit(x_train_scaled, y_train)

y_pred = xgb.predict(x_test_scaled)

print(classification_report(y_test, y_pred))

print("Accuracy:", metrics.accuracy_score(y_test,y_pred))
print("Precision:", metrics.precision_score(y_test,y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))

print("Training Score:",xgb.score(x_train,y_train))
print("Testing Score:",xgb.score(x_test,y_test))
