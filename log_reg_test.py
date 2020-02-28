# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:52:02 2019

@author: hari4
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset\
dataset = pd.read_csv("Social_Network_Ads.csv")
x_mtx = dataset.iloc[:, 2:-1].values
y_vctr = dataset.iloc[:, -1].values

#splitting data set into training set and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_mtx, y_vctr, test_size=0.25, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler

x_scl = StandardScaler()
x_train = x_scl.fit_transform(x_train)
x_test = x_scl.transform(x_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

y_prdc = classifier.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_prdc)

#Data Visualization(Training set)
from matplotlib.colors import ListedColormap

x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(x_set[:, 0].min() - 1, x_set[:, 0].max() + 1, 0.01),
                     np.arange(x_set[:, 1].min() - 1, x_set[:, 1].max() + 1, 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                color=ListedColormap(('yellow', 'blue'))(i), label=j)
plt.title("Logistic Regression (Training Set)")
plt.xlabel("Age")
plt.ylabel("Expected Salary")
plt.legend()
plt.show()

#Data Visualization Test Set
x_smpl, y_smpl = x_test, y_test
xa, xs = np.meshgrid(np.arange(x_smpl[:, 0].min() - 1, x_smpl[:, 0].max() + 1, 0.01),
                     np.arange(x_smpl[:, 1].min() - 1, x_smpl[:, 1].max() + 1, 0.01))
plt.contourf(xa, xs, classifier.predict(np.array([xa.ravel(), xs.ravel()]).T).reshape(xa.shape),
            alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(xa.min(), xa.max())
plt.ylim(xs.min(), xs.max())
for a, b in enumerate(np.unique(y_smpl)):
    plt.scatter(x_smpl[y_smpl == b, 0], x_smpl[y_smpl == b, 1], 
                color=ListedColormap(('yellow', 'blue'))(a), label=b)
plt.title("Logistic Regression (Test Set)")
plt.xlabel("Age")
plt.ylabel("Expected Salary")
plt.legend()
plt.show()