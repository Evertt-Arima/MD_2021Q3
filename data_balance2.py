# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 12:59:08 2021

@author: Desenvolvedor

Cria DataFrames balanceados para
treinamento e teste de classificadores.
"""

import numpy as np
import pandas as pd
import glob as gb



def get_train_data(genuine, features, rnd, session):
    """
    Parameters
    ----------
    genuine : TYPE int
        genuine user data
    features : TYPE string (fft, fft_full, Nickel, Kwapisz, raw, raw_rv, raw_mn)
        user features for train

    Returns
    -------
    train_data : TYPE pandas DataFrame
        DataFrame with all Genuine User features (user === 1) balanced with
            Impostor User features Sample (user === 0)
    """
    if session == 1:
        if features == "fft":
            path_session = gb.glob("relief_feat_selections/FFT5_features/session1/user*.csv")
        elif features == "Nickel" :
            path_session = gb.glob("relief_feat_selections/user_nickel/session1/user*.csv")
        elif features == "Kwapisz" :
            path_session = gb.glob("relief_feat_selections/user_kwapisz/session1/user*.csv")
    if session == 1:
        if features == "fft":
            path_session2 = gb.glob("feature_extraction/features/FFT5_features/session1/user*.csv")
        elif features == "Nickel" :
            path_session2 = gb.glob("feature_extraction/features/user_nickel/session1/user*.csv")
        elif features == "Kwapisz" :
            path_session2 = gb.glob("feature_extraction/features/user_kwapisz/session1/user*.csv")
        

        
    data_imp_general = pd.DataFrame()
    count = 0
    
    for u in range(0,len(path_session)):
        file = path_session[u]
        file2 = path_session2[u]
        
        if u == genuine:
            data_gen = pd.read_csv(file, sep=",", header=0)
            data_gen["user"] = 1
        else:
            aux = pd.read_csv(file2, sep=",", header=0)
            if count == 0:
                data_imp_general = aux
                count += 1
            else:
                data_imp_general = pd.concat([data_imp_general, aux], sort=False)
            

    df_spl = data_imp_general.sample(len(data_gen), random_state=rnd)
    #df_spl = data_imp_general.sample(len(data_gen))
    df_spl["user"] = 0
    gen_cols = data_gen.columns
    for col in df_spl.columns:
        if col not in gen_cols:
            df_spl.pop(col)
    
    train_data = pd.concat([data_gen, df_spl])
    return train_data 


def get_test_data(genuine, features, session):
    """
    Parameters
    ----------
    genuine : TYPE int
        genuine user data for test
    features : TYPE string (fft, fft_full, Nickel, Kwapisz, raw, raw_rv, raw_mn)
        choice of user features for test

    Returns
    -------
    test_data_genuine : TYPE pandas DataFrame
        DataFrame with genuine User features
    test_data_impostor : TYPE pandas DataFrame
        DataFrame with impostor User features
    """    
    if session == 2:
        if features == "fft":
            path_session = gb.glob("feature_extraction/features/FFT5_features/session2/user*.csv")
        elif features == "Nickel" :
            path_session = gb.glob("feature_extraction/features/user_nickel/session2/user*.csv")
        elif features == "Kwapisz" :
            path_session = gb.glob("feature_extraction/features/user_kwapisz/session2/user*.csv")
        
    
    cont = 0
    for u in range(0, len(path_session)):
        file = path_session[u]
        if u == genuine:
            test_data_genuine = pd.read_csv(file, sep=",", header=0)
        else:
            if cont == 0:
                test_data_impostor = pd.read_csv(file, sep=",", header=0)
                cont +=1
            else:
                aux = pd.read_csv(file, sep=",", header=0)
                test_data_impostor = test_data_impostor.append(aux, ignore_index=True, sort=False)

    return test_data_genuine, test_data_impostor
    



def rand_seed_gen(qty):
    np.random.seed(1978)
    rnd = list(np.random.random(size = qty) * 10000)
    rnd =  list(map(lambda x : int(x), rnd))
    return rnd





def cl_prep(user, feature_choice, times, train_session, test_session):
    
    seeds_list = rand_seed_gen(times)
    
    X_train = []
    y_train = []
    X_test_g = []
    y_test_g = []
    X_test_i = []
    y_test_i = []
    
    for rnd_sd in seeds_list:
        rnd = rnd_sd
        train = get_train_data(user, feature_choice, rnd, train_session)
        test_g, test_i = get_test_data(user, feature_choice, test_session)
        
        # if feature_choice != "raw":
        #     train = train.drop(columns = 'window')
        #     test_g = test_g.drop(columns = 'window')
        #     test_i = test_i.drop(columns = 'window')
        
        test_g["user"] = 1
        test_i["user"] = 0
        
        
        fea_cols = train.columns
        for col in test_g.columns:
            if col not in fea_cols:
                test_g.pop(col)
                
        for col in test_i.columns:
            if col not in fea_cols:
                test_i.pop(col)
        
        y_train.append(train.pop("user"))
        X_train.append(train)
        y_test_g.append(test_g.pop("user"))
        X_test_g.append(test_g)
        y_test_i.append(test_i.pop("user"))
        X_test_i.append(test_i)

    
    

    return X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i


#X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i = cl_prep(0, 'fft', 1, 1, 2)

'''
times = 1
session = 1
genuine = 0
features = 'fft'
rnd = rand_seed_gen(times)
train_session = 1
user = 1
'''