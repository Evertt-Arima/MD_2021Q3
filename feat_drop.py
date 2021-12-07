# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 10:25:26 2021

@author: Desenvolvedor
"""
import pandas as pd
import os
import data_balance as db
import numpy as np


def get_rel_feat(df, usr):
    df_rel = pd.DataFrame(df.iloc[usr])
    df_rel.sort_values(by=[df_rel.columns[0]], ascending=False, inplace=True)
    rel_feat = []
    for j in range(int(len(list(df_rel.index.values))/2)):
        rel_feat.append(list(df_rel.index.values)[j])
    return rel_feat


def get_user_file(usr):
    if usr < 10:
        usr_file = 'user0' + str(usr) + '.csv'
    else:
        usr_file = 'user' + str(usr) + '.csv'
    return usr_file


def get_feat(file):
    fea = file.split(sep=('_'))[0]
    return fea


def get_dest_path(session, features):
    if session == 1:
        if features == "fft":
            path_session = "relief_feat_selections/FFT5_features/session1/"
        elif features == "Nickel" :
            path_session = "relief_feat_selections/user_nickel/session1/"
        elif features == "Kwapisz" :
            path_session = "relief_feat_selections/user_kwapisz/session1/"
        
    else:
        if features == "fft":
            path_session = "relief_feat_selections/FFT5_features/session2/"
        elif features == "Nickel" :
            path_session = "relief_feat_selections/user_nickel/session2/"
        elif features == "Kwapisz" :
            path_session = "relief_feat_selections/user_kwapisz/session2/"
    return path_session


def get_feat_path(session, features):
    if session == 1:
        if features == "fft":
            path_session = "feature_extraction/features/FFT5_features/session1/"
        elif features == "Nickel" :
            path_session = "feature_extraction/features/user_nickel/session1/"
        elif features == "Kwapisz" :
            path_session = "feature_extraction/features/user_kwapisz/session1/"
        
    else:
        if features == "fft":
            path_session = "feature_extraction/features/FFT5_features/session2/"
        elif features == "Nickel" :
            path_session = "feature_extraction/features/user_nickel/session2/"
        elif features == "Kwapisz" :
            path_session = "feature_extraction/features/user_kwapisz/session2/"
    return path_session



    
path = 'Relief_test/'

files = os.listdir(path)



for file in files:    
    df = pd.read_csv(path+file)    
    feature = get_feat(file)   
    for i in range(len(df)):
        #i = 1
        user_file = get_user_file(i)
        rel_feat = get_rel_feat(df, i)
        df_f = pd.DataFrame()
        for feat in rel_feat:
            path_sess=get_feat_path(1, feature)
            df2 = pd.read_csv(path_sess+user_file)
            df_f[feat] = df2.pop(feat)
        path_dest = get_dest_path(1, feature)
        df_f.to_csv(path_dest+user_file, index=False)
       
    
    








'''




'''


















