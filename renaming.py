# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 09:34:22 2021

@author: Desenvolvedor
"""
import pandas as pd
import os
import data_balance as db


def prep_data(u, feat, times):
    X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i = db.cl_prep(u, feat, times, 1, 2)
    Xtrain = X_train[0].reset_index(drop=True)
    ytrain = y_train[0].reset_index(drop=True)  
    Xtest = pd.concat([X_test_g[0], X_test_i[0]], axis=0, ignore_index=True)
    ytest = pd.concat([y_test_g[0], y_test_i[0]], axis=0, ignore_index=True)
    return Xtrain, ytrain, Xtest, ytest


path = 'feature_extraction/features/FFT5_features/session1/'
path2 = 'Relief_feat_relev/'

files = os.listdir(path)
rl_files = os.listdir(path2)

times=1

features = ['fft', 'Nickel', 'Kwapisz']


for feat in features:
    Xtrain, ytrain, Xtest, ytest = prep_data(1, feat, times)
    cols = Xtrain.columns
    print(cols)
    for rl in rl_files:
        if feat in rl:
            df = pd.read_csv(path2+rl)
            df.columns = cols
            df.to_csv('Relief_test/'+feat+'_relief_scores.csv', index=False)


    