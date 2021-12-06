# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 22:09:16 2021

@author: Desenvolvedor

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from ReliefF import ReliefF

digits = load_digits(2)
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

fs = ReliefF(n_neighbors=100, n_features_to_keep=5)
X_train = fs.fit_transform(X_train, y_train)
X_test_subset = fs.transform(X_test)
print(X_test.shape, X_test_subset.shape)



"""
#+++++++++++++++++++++++++++++++++++++++


from ReliefF import ReliefF
import numpy as np
from sklearn import datasets
import pandas as pd

#example of 2 class problem
data = np.array([[9,2,2],[5,1,0],[9,3,2],[8,3,1],[6,0,0]])
target = np.array([0,0,1,1,1])

fs = ReliefF(n_neighbors=3, n_features_to_keep=2)
X_train = fs.fit_transform(data, target)
print(X_train)
print("--------------")
print("(No. of tuples, No. of Columns before ReliefF) : "+str(data.shape)+
      "\n(No. of tuples , No. of Columns after ReliefF) : "+str(X_train.shape))


#example of multi class problem
iris = datasets.load_iris()
X = iris.data
Y = iris.target

fs = ReliefF(n_neighbors=20, n_features_to_keep=2)
X_train = fs.fit_transform(X, Y)
print("(No. of tuples, No. of Columns before ReliefF) : "+str(iris.data.shape)+
      "\n(No. of tuples, No. of Columns after ReliefF) : "+str(X_train.shape))




import glob as gb

import data_balance as db

X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i = db.cl_prep(0, "Nickel", times, 1, 2)

Xtrain = X_train[0].reset_index(drop=True)
ytrain = y_train[0].reset_index(drop=True)

X = np.array(Xtrain)
Y = np.array(ytrain)
keep = int(np.round(len(Xtrain.columns)/2))
fs = ReliefF(n_neighbors=20, n_features_to_keep=10)
X_train_ReliefF = fs.fit_transform(X, Y)

print("(No. of tuples, No. of Columns before ReliefF) : "+str(Xtrain.shape)+
      "\n(No. of tuples, No. of Columns after ReliefF) : "+str(X_train_ReliefF.shape))




Xtest = pd.concat([X_test_g[0], X_test_i[0]], axis=0, ignore_index=True)
ytest = pd.concat([y_test_g[0], y_test_i[0]], axis=0, ignore_index=True)




score = fs.feature_scores()

















