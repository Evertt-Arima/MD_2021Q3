# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:59:14 2021

@author: Desenvolvedor
"""
# Base Libraries
import numpy as np 
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import os
# Transformation
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import power_transform
from sklearn.pipeline import Pipeline
# Feature Selection
import sklearn_relief as sr
# Models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import data_balance as db


def prep_data(u, feat, times):
    X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i = db.cl_prep(u, feat, times, 1, 2)
    Xtrain = X_train[0].reset_index(drop=True)
    ytrain = y_train[0].reset_index(drop=True)  
    Xtest = pd.concat([X_test_g[0], X_test_i[0]], axis=0, ignore_index=True)
    ytest = pd.concat([y_test_g[0], y_test_i[0]], axis=0, ignore_index=True)
    return Xtrain, ytrain, Xtest, ytest

def X_y_prep(Xtrain, ytrain):
    X = np.array(Xtrain)
    y = np.array(ytrain)
    return X, y

def split_data(X, y):
    X_train_R, X_test_R, y_train_R, y_test_R = train_test_split(X,y, test_size = 0.2, random_state = 0)
    return X_train_R, X_test_R, y_train_R, y_test_R

def relief_score_RFR(n, X_train_R, X_test_R, y_train_R, y_test_R):
    fs = sr.RReliefF(n_features = n+1)
    relief = Pipeline([('fs', fs), ('m', RandomForestRegressor())])
    #relief = Pipeline([('fs', fs), ('m', KNeighborsRegressor(n_neighbors=1))])
    relief.fit(X_train_R,y_train_R)
    score = relief.score(X_test_R,y_test_R)
    return score

def relief_score_kNN(n, X_train_R, X_test_R, y_train_R, y_test_R):
    fs = sr.RReliefF(n_features = n+1)
    #relief = Pipeline([('fs', fs), ('m', RandomForestRegressor())])
    relief = Pipeline([('fs', fs), ('m', KNeighborsRegressor(n_neighbors=3))])
    relief.fit(X_train_R,y_train_R)
    score = relief.score(X_test_R,y_test_R)
    return score

def relief_fit(X_train_R, X_test_R, y_train_R, y_test_R):
    r = sr.RReliefF(n_features = n_o_feat)
    new_order = r.fit_transform(X_train_R,y_train_R)
    print(new_order)
    return new_order

def calc_score(X_train_R, X_test_R, y_train_R, y_test_R):
    nof_list=np.arange(1, n_o_feat+1)
    #nof_list=np.array(cols)
    high_score=0
    nof=0           
    score_list =[]
    
    for n in range(len(nof_list)):
        print(n, "/", len(nof_list))
        score = relief_score_kNN(n, X_train_R, X_test_R, y_train_R, y_test_R)
        score_list.append(score)
        print(f'NOF: {nof_list[n]}, Score: {score}')
        '''
        if(score > high_score):
            high_score = score
            nof = nof_list[n]
        print(f'High Score: NOF: {nof}, Score: {high_score}\n')
        '''
    return score_list



times=1

features = ['fft', 'Nickel', 'Kwapisz', 'fft_full']

path = 'feature_extraction/features/FFT5_features/session1/'

files = os.listdir(path)


feat = features[1]
sc = pd.DataFrame()
for u in range(len(files)):
    print (f'User: {u}, Feature: {feat}')

    Xtrain, ytrain, Xtest, ytest = prep_data(u, feat, times)
    X, y = X_y_prep(Xtrain, ytrain)
    n_o_feat = len(Xtrain.columns)
    cols = Xtrain.columns
    X_train_R, X_test_R, y_train_R, y_test_R = split_data(X, y)
    
    print("\nInitiating Score Loop!!!")
    score_list = calc_score(X_train_R, X_test_R, y_train_R, y_test_R)
    
    print(score_list)
    sc1 = pd.DataFrame(score_list).transpose()
    sc=pd.concat([sc, sc1], axis=0)

    #sc.columns = Xtrain.columns

    sc.to_csv('Relief_feat_relev/'+feat+'_relief_scores.csv', index=False)


'''
for feat in features:
    sc = pd.DataFrame()
    for u in range(len(files)):
        print (f'User: {u}, Feature: {feat}')

        Xtrain, ytrain, Xtest, ytest = prep_data(u, feat, times)
        X, y = X_y_prep(Xtrain, ytrain)
        n_o_feat = len(Xtrain.columns)
        cols = Xtrain.columns
        X_train_R, X_test_R, y_train_R, y_test_R = split_data(X, y)
        
        print("\nInitiating Score Loop!!!")
        score_list = calc_score(X_train_R, X_test_R, y_train_R, y_test_R)
        
        print(score_list)
        sc1 = pd.DataFrame(score_list).transpose()
        sc=pd.concat([sc, sc1], axis=0)

    sc.columns = Xtrain.columns
    
    sc.to_csv('Relief_feat_relev/'+feat+'_relief_scores.csv', index=False)
    '''