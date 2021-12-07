# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:21:52 2021

@author: Desenvolvedor
"""

import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn import metrics

import numpy as np
#from sklearn.ensemble import RandomForestClassifier
from data_balance import *
from numpy.random import SeedSequence
import random



def dt_clf(user, feature, X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i):
    """
    Parameters
    ----------
    user : TYPE int
        genuine user
    feature_choice : TYPE string (fft, fft_full, Nickel, Kwapisz, raw, raw_rv, raw_mn)
        choice of user features for test

    Returns
    -------
    prediction_genuine : TYPE Array
        DataFrame with genuine User features
    prediction_impostor : TYPE Array
        DataFrame with impostor User features
        
    n_estimator: set to 100 according to
        'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20forest#sklearn.ensemble.RandomForestClassifier'
    """  
    
    #dtree = DecisionTreeClassifier()
    #dtree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    dtree = DecisionTreeClassifier(criterion="entropy")

    dtree = dtree.fit(X_train, y_train)
    
    features = X_train.columns
    data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
    graph = pydotplus.graph_from_dot_data(data)
    graph_name = 'GraphicsRLF/' + feature +'/user_' + str(user) + '.png'
    print(graph_name)
    graph.write_png(graph_name)
    
    img=pltimg.imread(graph_name)
    imgplot = plt.imshow(img)
    plt.show()


    prediction_genuine = dtree.predict(X_test_g)
    prediction_impostor = dtree.predict(X_test_i)
    

    
    return prediction_genuine, prediction_impostor
'''
#feature_extraction/features/user_nickel/session1

def rf_clf(X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i):
    """
    Parameters
    ----------
    user : TYPE int
        genuine user
    feature_choice : TYPE string (fft, fft_full, Nickel, Kwapisz, raw, raw_rv, raw_mn)
        choice of user features for test

    Returns
    -------
    prediction_genuine : TYPE Array
        DataFrame with genuine User features
    prediction_impostor : TYPE Array
        DataFrame with impostor User features
        
    n_estimator: set to 100 according to
        'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20forest#sklearn.ensemble.RandomForestClassifier'
    """  
    
        
    clf = RandomForestClassifier(n_estimators=100, random_state=1)
    
    clf.fit(X_train, y_train)
    
    # ft_importance = clf.feature_importances_

    prediction_genuine = clf.predict(X_test_g)
    prediction_impostor = clf.predict(X_test_i)
    
    return prediction_genuine, prediction_impostor


'''