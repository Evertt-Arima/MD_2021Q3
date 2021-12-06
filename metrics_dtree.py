# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:39:22 2020

@author: Arima

Gera um arquivo CSV com a Acurácia Balanceada
resultante das predições dos classificadores.

Precisa ser corrigido os laços de repetição,
atualmente fixo em 50 usuários.
"""

import glob as gb
import numpy as np
import pandas as pd
#from sklearn.ensemble import RandomForestClassifier
from decision_tree import dt_clf
import data_balance as db
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
        prediction_genuine =[]
        prediction_impostor = []
        
        for j in range (0, len(X_train)):
            pre_g, pre_i = dt_clf(i, 'Nickel', X_train[j], y_train[j], X_test_g[j], y_test_g[j], X_test_i[j], y_test_i[j])
            prediction_genuine.append(pre_g)
            prediction_impostor.append(pre_i)
            print(i, j)
            print("Accuracy genuine:",metrics.accuracy_score(y_test_g[j], pre_g))
            print("Accuracy impostor:",metrics.accuracy_score(y_test_i[j], pre_i), '\n')
        
            
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
        prediction_genuine =[]
        prediction_impostor = []
        
        for j in range (0, len(X_train)):
            pre_g, pre_i = dt_clf(i, 'FFT', X_train[j], y_train[j], X_test_g[j], y_test_g[j], X_test_i[j], y_test_i[j])
            prediction_genuine.append(pre_g)
            prediction_impostor.append(pre_i)
            
            
        bal_acc_f, fnmr_a, fmr_a, std_a = metricas(prediction_genuine, prediction_impostor)
        fft_bacc.append(bal_acc_f)
        fnmr.append(fnmr_a)
        fmr.append(fmr_a)
        stdv.append(std_a)
        
    return pd.DataFrame(fft_bacc, columns = ['FFT 5 BAcc']), pd.DataFrame(fnmr, columns = ['FFT 5 FNMR']), pd.DataFrame(fmr, columns = ['FFT 5 FMR']), pd.DataFrame(stdv, columns = ['FFT 5 STD'])



def fft_full (times):
    fft_f_bacc = []
    fnmr = []
    fmr = []
    stdv = []
    
    for i in range (0, size):
        X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i = db.cl_prep(i, "fft_full", times, 1, 2)
        prediction_genuine =[]
        prediction_impostor = []
        
        for j in range (0, len(X_train)):
            pre_g, pre_i = dt_clf(i, 'FFT_full', X_train[j], y_train[j], X_test_g[j], y_test_g[j], X_test_i[j], y_test_i[j])
            prediction_genuine.append(pre_g)
            prediction_impostor.append(pre_i)
            
            
        bal_acc_ff, fnmr_a, fmr_a, std_a = metricas(prediction_genuine, prediction_impostor)
        fft_f_bacc.append(bal_acc_ff)
        fnmr.append(fnmr_a)
        fmr.append(fmr_a)
        stdv.append(std_a)

        
    return pd.DataFrame(fft_f_bacc, columns = ['FFT full BAcc']), pd.DataFrame(fnmr, columns = ['FFT full FNMR']), pd.DataFrame(fmr, columns = ['FFT full FMR']), pd.DataFrame(stdv, columns = ['FFT full STD'])



def kwapisz (times):
    kwap_bacc = []
    fnmr = []
    fmr = []
    stdv = []
    
    for i in range (0, size):
        X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i = db.cl_prep(i, "Kwapisz", times, 1, 2)
        prediction_genuine =[]
        prediction_impostor = []
        
        for j in range (0, len(X_train)):
            pre_g, pre_i = dt_clf(i, 'Kwapisz', X_train[j], y_train[j], X_test_g[j], y_test_g[j], X_test_i[j], y_test_i[j])
            prediction_genuine.append(pre_g)
            prediction_impostor.append(pre_i)
            
            
        bal_acc_k, fnmr_a, fmr_a, std_a = metricas(prediction_genuine, prediction_impostor)
        kwap_bacc.append(bal_acc_k)
        fnmr.append(fmr_a)
        fmr.append(fmr_a)
        stdv.append(std_a)
        
    return pd.DataFrame(kwap_bacc, columns = ['Kwapisz BAcc']), pd.DataFrame(fnmr, columns = ['Kwapisz FNMR']), pd.DataFrame(fmr, columns = ['Kwapisz FMR']), pd.DataFrame(stdv, columns = ['Kwapisz STD'])



def raw (times):
    raw_bacc = []
    fnmr = []
    fmr = []
    stdv = []
       
            
    for i in range (0, size):
        X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i = db.cl_prep(i, "raw", times, 1, 2)
        prediction_genuine =[]
        prediction_impostor = []
        
        for j in range (0, len(X_train)):
            pre_g, pre_i = dt_clf(i, 'Raw', X_train[j], y_train[j], X_test_g[j], y_test_g[j], X_test_i[j], y_test_i[j])
            prediction_genuine.append(pre_g)
            prediction_impostor.append(pre_i)
            
            
        bal_acc_r, fnmr_a, fmr_a, std_a = metricas(prediction_genuine, prediction_impostor)
        raw_bacc.append(bal_acc_r)
        fnmr.append(fnmr_a)
        fmr.append(fmr_a)
        stdv.append(std_a)
    
    return pd.DataFrame(raw_bacc, columns = ['Raw BAcc']), pd.DataFrame(fnmr, columns = ['Raw FNMR']), pd.DataFrame(fmr, columns = ['Raw FMR']), pd.DataFrame(stdv, columns = ['Raw STD'])



def raw_resultant (times):
    raw_rv_bacc = []
    fnmr = []
    fmr = []
    stdv = []
    
    for i in range (0, size):
       X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i = db.cl_prep(i, "raw_rv", times, 1, 2)
       prediction_genuine =[]
       prediction_impostor = []
       
       for j in range (0, len(X_train)):
           pre_g, pre_i = dt_clf(i, 'Raw_VR', X_train[j], y_train[j], X_test_g[j], y_test_g[j], X_test_i[j], y_test_i[j])
           prediction_genuine.append(pre_g)
           prediction_impostor.append(pre_i)
           
           
       bal_acc_r_rv, fnmr_a, fmr_a, std_a = metricas(prediction_genuine, prediction_impostor)
       raw_rv_bacc.append(bal_acc_r_rv)
       fnmr.append(fnmr_a)
       fmr.append(fmr_a)
       stdv.append(std_a)
    
    
    return pd.DataFrame(raw_rv_bacc, columns = ['Raw Resultant BAcc']), pd.DataFrame(fnmr, columns = ['Raw Resultant FNMR']), pd.DataFrame(fmr, columns = ['Raw Resultant FMR']), pd.DataFrame(stdv, columns = ['Raw Resultant STD'])



def raw_means (times):
    raw_mn_bacc = []
    fnmr = []
    fmr = []
    stdv = []
    
    for i in range (0, size):
       X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i = db.cl_prep(i, "raw_mn", times, 1, 2)
       prediction_genuine =[]
       prediction_impostor = []
       
       for j in range (0, len(X_train)):
           pre_g, pre_i = dt_clf(i, 'Raw_means', X_train[j], y_train[j], X_test_g[j], y_test_g[j], X_test_i[j], y_test_i[j])
           prediction_genuine.append(pre_g)
           prediction_impostor.append(pre_i)
           
           
       bal_acc_r_mn, fnmr_a, fmr_a, std_a = metricas(prediction_genuine, prediction_impostor)
       raw_mn_bacc.append(bal_acc_r_mn)
       fnmr.append(fnmr_a)
       fmr.append(fmr_a)
       stdv.append(std_a)
   
        
    return pd.DataFrame(raw_mn_bacc, columns = ['Raw Means BAcc']), pd.DataFrame(fnmr, columns = ['Raw Means FNMR']), pd.DataFrame(fmr, columns = ['Raw Means FMR']), pd.DataFrame(stdv, columns = ['Raw Means STD'])



def metrics_file (times):
    u = []

    for i in range (0, size):
        u.append(i)
    us = pd.DataFrame(u, columns = ["user"])
     
    nicke, fnmr_n, fmr_n, std_n = nickel(times)
    kwapis, fnmr_k, fmr_k, std_k = kwapisz(times)
    ff, fnmr_f5, fmr_f5, std_f5 = fft(times)
    fft_ful, fnmr_ff, fmr_ff, std_ff = fft_full(times)
    #rw, fnmr_r, fmr_r, std_r = raw(times)
    #rw_rv, fnmr_rrv, fmr_rrv, std_rrv = raw_resultant(times) 
    #rw_mn, fnmr_rm, fmr_rm, std_rm = raw_means(times)
    
        
    #df = pd.concat([us, ff, fft_ful, nicke, kwapis, rw, rw_rv, rw_mn], axis=1, sort=False)
    #fnmr = pd.concat([us, fnmr_f5, fnmr_ff, fnmr_n, fnmr_k, fnmr_r, fnmr_rrv, fnmr_rm], axis=1, sort=False)
    #fmr = pd.concat([us, fmr_f5, fmr_ff, fmr_n, fmr_k, fmr_r, fmr_rrv, fmr_rm], axis=1, sort=False)
    #stdv = pd.concat([us, std_f5, std_ff, std_n, std_k, std_r, std_rrv, std_rm], axis=1, sort=False)
    
    df = pd.concat([us, ff, fft_ful, nicke, kwapis], axis=1, sort=False)
    fnmr = pd.concat([us, fnmr_f5, fnmr_ff, fnmr_n, fnmr_k], axis=1, sort=False)
    fmr = pd.concat([us, fmr_f5, fmr_ff, fmr_n, fmr_k], axis=1, sort=False)
    stdv = pd.concat([us, std_f5, std_ff, std_n, std_k], axis=1, sort=False)
    
    #print (df)
    
    df.to_csv("results_bacc/metrics_1_2.csv", index=False)

    return df, fnmr, fmr, stdv
    #return nicke, fnmr_n, fmr_n, std_n

path_session = gb.glob("feature_extraction/features/user_raw/session1/user*.csv")
size = len(path_session)
times = 1

df_f, fnmr, fmr, stdv = metrics_file(times)


fmr.to_csv("data/data extracted/fmr_1_2.csv", index=False)
fnmr.to_csv("data/data extracted/fnmr_1_2.csv", index=False)
stdv.to_csv("data/data extracted/stdv_1_2.csv", index=False)



# nickel = nickel()
#kwapisz = kwapisz()
# fft = fft()
# fft_full = fft_full()
# raw = raw()
# raw_rv = raw_resultant() 
# raw_mn = raw_means()
    
'''

fft_bacc = []
fft_f_bacc = []
nick_bacc = []
kwap_bacc = []
raw_bacc = []
raw_rv_bacc = []
raw_mn_bacc = []
u = []

for i in range (0, 50):
    u.append(i)
    gen_f, imp_f = rf_clf (i, "fft")
    bal_acc_f = metrics(gen_f, imp_f)
    fft_bacc.append(bal_acc_f)
for i in range (0, 50):
    gen_ff, imp_ff = rf_clf (i, "fft_full")
    bal_acc_ff = metrics(gen_ff, imp_ff)
    fft_f_bacc.append(bal_acc_ff)
for i in range (0, 50):
    gen_n, imp_n = rf_clf (i, "Nickel")
    bal_acc_n = metrics(gen_n, imp_n)
    nick_bacc.append(bal_acc_n)
for i in range (0, 50):
    gen_k, imp_k = rf_clf (i, "Kwapisz")
    bal_acc_k = metrics(gen_k, imp_k)
    kwap_bacc.append(bal_acc_k)
for i in range (0, 50):    
    gen_r, imp_r = rf_clf (i, "raw")
    bal_acc_r = metrics(gen_r, imp_r)
    raw_bacc.append(bal_acc_r)
for i in range (0, 50):    
    gen_r_rv, imp_r_rv = rf_clf (i, "raw_rv")
    bal_acc_r_rv = metrics(gen_r_rv, imp_r_rv)
    raw_rv_bacc.append(bal_acc_r_rv)
for i in range (0, 50):    
    gen_r_mn, imp_r_mn = rf_clf (i, "raw_mn")
    bal_acc_r_mn = metrics(gen_r_mn, imp_r_mn)
    raw_mn_bacc.append(bal_acc_r_mn)
    
# print (fft_bacc)
# print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# print (nick_bacc)




df = pd.concat([(pd.DataFrame(u)),
                (pd.DataFrame(nick_bacc)),
                (pd.DataFrame(kwap_bacc)),                
                (pd.DataFrame(fft_bacc)),
                (pd.DataFrame(fft_f_bacc)),
                (pd.DataFrame(raw_bacc)),
                (pd.DataFrame(raw_rv_bacc)),
                (pd.DataFrame(raw_mn_bacc)),], axis=1, sort=False)

df.columns = ['User', 'Nickel BAcc', 'FFT 5 BAcc', 'FFT full BAcc', 'Raw BAcc', 'Raw Resultant BAcc', 'Raw Means BAcc']

print (df)

df.to_csv("metrics.csv", index=False)


fnmr = []    

for i in prediction_genuine:
    aux = 1.0 - sum(i) / len(i)
    fnmr.append(aux)

fmr = []

for i in prediction_impostor:
    aux = sum(i) / len(i)
    fmr.append(aux)    

#fmr = sum(prediction_impostor) / len(prediction_impostor)
#fnmr = 1.0 - sum(prediction_genuine) / len(prediction_genuine)
#bacc = 1.0 - (fmr + fnmr) / 2.0


mean_gen = np.mean(fnmr)
mean_imp = np.mean(fmr)

bacc = 1.0 - (mean_imp + mean_gen) / 2.0



X_train, y_train, X_test_g, y_test_g, X_test_i, y_test_i = cl_prep(1, "Kwapisz", times)


prediction_genuine =[]
prediction_impostor = []

print (len(X_train))
for i in range (0, len(X_train)):
    pre_g, pre_i = rf_clf(X_train[i], y_train[i], X_test_g[i], y_test_g[i], X_test_i[i], y_test_i[i])
    prediction_genuine.append(pre_g)
    prediction_impostor.append(pre_i)





'''