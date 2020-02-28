# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 23:07:29 2019

@author: hari4
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
x_mtx = dataset.iloc[:, 2:-1].values
y_vctr = dataset.iloc[:, -1].values

#splitting dataset into training set and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_mtx, y_vctr, test_size=0.25, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler

x_scl = StandardScaler()
x_train = x_scl.fit_transform(x_train)
x_test = x_scl.transform(x_test)

#Decision Tree Classifier classifier
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)

y_prdc = classifier.predict(x_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_prdc)

#Data visualization Training set
from matplotlib.colors import ListedColormap
x_lrn, y_lrn = x_train, y_train
xl1, xl2 = np.meshgrid(np.arange(x_lrn[:, 0].min() - 1, x_lrn[:, 0].max() + 1, 0.01),
                       np.arange(x_lrn[:, 1].min() - 1, x_lrn[:, 1].max() + 1, 0.01))
plt.contourf(xl1, xl2, classifier.predict(np.array([xl1.ravel(), xl2.ravel()]).T).reshape(xl1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(xl1.min(), xl1.max())
plt.ylim(xl2.min(), xl2.max())
for i, j in enumerate(np.unique(y_lrn)):
    plt.scatter(x_lrn[y_lrn == j, 0], x_lrn[y_lrn == j, 1], 
                color=ListedColormap(('yellow', 'blue'))(i), label=j)
plt.title("Decision Tree Classifier (Training Set)")
plt.xlabel("Age")
plt.ylabel("Expected Salary")
plt.legend()
plt.show()

#Data visualization Test set
x_smpl, y_smpl = x_test, y_test
xs1, xs2 = np.meshgrid(np.arange(x_smpl[:, 0].min() - 1, x_smpl[:, 0].max() + 1, 0.01),
                       np.arange(x_smpl[:, 1].min() - 1, x_smpl[:, 1].max() + 1, 0.01))
plt.contourf(xs1, xs2, classifier.predict(np.array([xs1.ravel(), xs2.ravel()]).T).reshape(xs1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(xs1.min(), xs1.max())
plt.ylim(xs2.min(), xs2.max())
for a, b in enumerate(np.unique(y_smpl)):
    plt.scatter(x_smpl[y_smpl == b, 0], x_smpl[y_smpl == b, 1],
                color=ListedColormap(('yellow', 'blue'))(a), label=b)
plt.title("Decision Tree Classifier (Test Set)")
plt.xlabel("Age")
plt.ylabel("Expected Salary")
plt.legend()
plt.show()    