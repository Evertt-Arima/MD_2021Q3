# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:12:54 2021

@author: Desenvolvedor

Gera um arquivo CSV com a Acurácia Balanceada
resultante das predições dos classificadores.


"""

import glob as gb
import numpy as np
import pandas as pd
#from sklearn.ensemble import RandomForestClassifier
from decision_tree2 import dt_clf
import data_balance2 as db
from sklearn import metrics

def metricas (prediction_genuine, prediction_impostor):
    """
    Parameters
    ----------
    prediction_genuine : TYPE Array 
        DataFrame with genuine User features
    prediction_impostor : TYPE Array
        DataFrame with impostor User features

    """  
        
    fnmr = []    
    for i in prediction_genuine:
        aux = 1.0 - sum(i) / len(i)
        fnmr.append(aux)
    
    fmr = []
    for i in prediction_impostor:
        aux = sum(i) / len(i)
        fmr.append(aux)    
    
    bacc_aux = []
    for i in range(0, len(fnmr)):
        aux = 1.0 - (fmr[i] + fnmr[i]) / 2.0
        bacc_aux.append(aux)
    
    mean_gen = np.mean(fnmr)
    mean_imp = np.mean(fmr)
    
    stdv = np.std(bacc_aux)
    
    bacc = 1.0 - (mean_imp + mean_gen) / 2.0
    
    #fmr = sum(prediction_impostor) / len(prediction_impostor)
    #fnmr = 1.0 - sum(prediction_genuine) / len(prediction_genuine)
    #bacc = 1.0 - (fmr + fnmr) / 2.0
    #bacc = 1.0 - HTER
    # print ("False Match Rate (FMR): " + str(fmr))
    # print ("False Non-Match Rate (FNMR): " + str(fnmr))
    # print ("Balanced Accuracy (BAcc): " + str(bacc))
    
    return bacc, mean_gen, mean_imp, stdv


# a, b = rf_clf (1, "fft")
# metrics(a, b)
# c, d = rf_clf (1, "Nickel")
# metrics(c, d)
    
def nickel (times):
    nick_bacc = []
    fnmr = []
    fmr = []
    stdv = []
        
    for i in range (0, size):
        X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i = db.cl_prep(i, "Nickel", times, 1, 2)
        X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i = X_train[0], y_train[0], X_test_g[0], y_test_g[0], X_test_i[0], y_test_i[0]
        prediction_genuine =[]
        prediction_impostor = []
        
    
        pre_g, pre_i = dt_clf(i, 'Nickel', X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i)
        prediction_genuine.append(pre_g)
        prediction_impostor.append(pre_i)
        #print(i, j)
        print("Accuracy genuine:",metrics.accuracy_score(y_test_g, pre_g))
        print("Accuracy impostor:",metrics.accuracy_score(y_test_i, pre_i), '\n')
    
            
        bal_acc_n, fnmr_a, fmr_a, std_a = metricas(prediction_genuine, prediction_impostor)
        nick_bacc.append(bal_acc_n)
        fnmr.append(fnmr_a)
        fmr.append(fmr_a)
        stdv.append(std_a)
    
    return pd.DataFrame(nick_bacc, columns = ['Nickel BAcc']), pd.DataFrame(fnmr, columns = ['Nickel FNMR']), pd.DataFrame(fmr, columns = ['Nickel FMR']), pd.DataFrame(stdv, columns = ['Nickel STD'])



def fft (times):
    fft_bacc = []
    fnmr = []
    fmr = []
    stdv = []
        
    for i in range (0, size):
        X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i = db.cl_prep(i, "fft", times, 1, 2)
        X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i = X_train[0], y_train[0], X_test_g[0], y_test_g[0], X_test_i[0], y_test_i[0]

        prediction_genuine =[]
        prediction_impostor = []
        
        
        pre_g, pre_i = dt_clf(i, 'FFT', X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i)
        prediction_genuine.append(pre_g)
        prediction_impostor.append(pre_i)
        
            
        bal_acc_f, fnmr_a, fmr_a, std_a = metricas(prediction_genuine, prediction_impostor)
        fft_bacc.append(bal_acc_f)
        fnmr.append(fnmr_a)
        fmr.append(fmr_a)
        stdv.append(std_a)
        
    return pd.DataFrame(fft_bacc, columns = ['FFT 5 BAcc']), pd.DataFrame(fnmr, columns = ['FFT 5 FNMR']), pd.DataFrame(fmr, columns = ['FFT 5 FMR']), pd.DataFrame(stdv, columns = ['FFT 5 STD'])




def kwapisz (times):
    kwap_bacc = []
    fnmr = []
    fmr = []
    stdv = []
    
    for i in range (0, size):
        X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i = db.cl_prep(i, "Kwapisz", times, 1, 2)
        X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i = X_train[0], y_train[0], X_test_g[0], y_test_g[0], X_test_i[0], y_test_i[0]

        prediction_genuine =[]
        prediction_impostor = []

        pre_g, pre_i = dt_clf(i, 'Kwapisz', X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i)
        prediction_genuine.append(pre_g)
        prediction_impostor.append(pre_i)
            
            
        bal_acc_k, fnmr_a, fmr_a, std_a = metricas(prediction_genuine, prediction_impostor)
        kwap_bacc.append(bal_acc_k)
        fnmr.append(fmr_a)
        fmr.append(fmr_a)
        stdv.append(std_a)
        
    return pd.DataFrame(kwap_bacc, columns = ['Kwapisz BAcc']), pd.DataFrame(fnmr, columns = ['Kwapisz FNMR']), pd.DataFrame(fmr, columns = ['Kwapisz FMR']), pd.DataFrame(stdv, columns = ['Kwapisz STD'])




def metrics_file (times):
    u = []

    for i in range (0, size):
        u.append(i)
    us = pd.DataFrame(u, columns = ["user"])
     
    nicke, fnmr_n, fmr_n, std_n = nickel(times)
    kwapis, fnmr_k, fmr_k, std_k = kwapisz(times)
    ff, fnmr_f5, fmr_f5, std_f5 = fft(times)
    #fft_ful, fnmr_ff, fmr_ff, std_ff = fft_full(times)
    #rw, fnmr_r, fmr_r, std_r = raw(times)
    #rw_rv, fnmr_rrv, fmr_rrv, std_rrv = raw_resultant(times) 
    #rw_mn, fnmr_rm, fmr_rm, std_rm = raw_means(times)
    
        
    #df = pd.concat([us, ff, fft_ful, nicke, kwapis, rw, rw_rv, rw_mn], axis=1, sort=False)
    #fnmr = pd.concat([us, fnmr_f5, fnmr_ff, fnmr_n, fnmr_k, fnmr_r, fnmr_rrv, fnmr_rm], axis=1, sort=False)
    #fmr = pd.concat([us, fmr_f5, fmr_ff, fmr_n, fmr_k, fmr_r, fmr_rrv, fmr_rm], axis=1, sort=False)
    #stdv = pd.concat([us, std_f5, std_ff, std_n, std_k, std_r, std_rrv, std_rm], axis=1, sort=False)
    
    df = pd.concat([us, ff, nicke, kwapis], axis=1, sort=False)
    fnmr = pd.concat([us, fnmr_f5, fnmr_n, fnmr_k], axis=1, sort=False)
    fmr = pd.concat([us, fmr_f5, fmr_n, fmr_k], axis=1, sort=False)
    stdv = pd.concat([us, std_f5, std_n, std_k], axis=1, sort=False)
    
    #print (df)
    
    df.to_csv("results_baccRLF/metrics_1_2.csv", index=False)

    return df, fnmr, fmr, stdv
    #return nicke, fnmr_n, fmr_n, std_n

size = len(gb.glob("relief_feat_selections/FFT5_features/session1/user*.csv"))
times = 1

df_f, fnmr, fmr, stdv = metrics_file(times)


fmr.to_csv("dataRLF/data extracted/fmr_1_2.csv", index=False)
fnmr.to_csv("dataRLF/data extracted/fnmr_1_2.csv", index=False)
stdv.to_csv("dataRLF/data extracted/stdv_1_2.csv", index=False)

