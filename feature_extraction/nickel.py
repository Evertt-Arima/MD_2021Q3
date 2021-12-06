#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:01:42 2020

@author: Arima

Este arquivo processa os dados
segundo os parâmetros do
trabalho de Nickel
"""
from user import *
from dataImport import *
import create_csv as cr_csv
import numpy as np
import pandas as pd
from statistic import *
import csv
import glob as gb



def histogram_nickel(data, box):
    '''
    Parameters
    ----------
    data : TYPE list
        DESCRIPTION
        windowed data (x, y, z, rv).
    box : TYPE int
        DESCRIPTION
        desired amount of bins for the data to be distributed.

    Returns
    -------
    df_hist : TYPE Dataframe
        DESCRIPTION
        distribution in amount of input bins.

    '''
    hist = []
    for i in range (len(data)):
        x = []
        y = []
        z = []
        rv = []
        aux = []
        for j in range(len(data[i])):
            x.append(data[i][j][0])
            y.append(data[i][j][1])
            z.append(data[i][j][2])
            rv.append(data[i][j][3])
        aux_x = np.asarray(np.histogram(x, bins=box))
        aux_x = aux_x[0]
        aux_y = np.asarray(np.histogram(y, bins=box))
        aux_y = aux_y[0]
        aux_z = np.asarray(np.histogram(z, bins=box))
        aux_z = aux_z[0]
        aux_rv = np.asarray(np.histogram(rv, bins=box))
        aux_rv = aux_rv[0]
        aux = np.append(aux_x, [aux_y, aux_z, aux_rv])
        hist.append(aux)
    df_hist = pd.DataFrame(hist)
    return df_hist



def creating_nickel_files(session, win, ovlap):

    sess = create_sess(session)
        
    for u in range(0, len(sess)):
        usr = create_user(u, get_user_data(u, session))
        #data = usr.data

        # print (data)

        c_data = center(usr.data)
        
        rv_data = resultant_vector(c_data)
        c_data["res"] = pd.concat([rv_data], axis=1, sort=False)
        
        # print (c_data)
        usr = create_user(u, c_data)
        
        #w_data = windowed_data(usr, window(win))
        w_data = windowed_data_overlap(usr, window(win), ovlap)
        
        sign_c = signal_change(w_data)
        media = mean_window(w_data) #OK
        standard_deviation = std_window(w_data)
        minimo = min_window(w_data)
        maximo = max_window(w_data)
        rms = root_mean_square(w_data)
        
        # Histogram
        df_hist = histogram_nickel(w_data, 10)
        
        wc = []

        xmean = []
        ymean = []
        zmean = []
        rvmean = []
        
        xstd = []
        ystd = []
        zstd = []
        rvstd = []
        
        xmin = []
        ymin = []
        zmin = []
        rvmin = []
        
        xmax = []
        ymax = []
        zmax = []
        rvmax = []
        
        xrms = []
        yrms = []
        zrms = []
        rvrms = []
        
        xsc = []
        ysc = []
        zsc = []
        rvsc = []
        
        user = []
        
        
        for i in range(0, len(w_data)):
            wc.append(i)

            xmean.append(media[i][0])
            ymean.append(media[i][1])
            zmean.append(media[i][2])
            rvmean.append(media[i][3])
            
            xstd.append(standard_deviation[i][0])
            ystd.append(standard_deviation[i][1])
            zstd.append(standard_deviation[i][2])
            rvstd.append(standard_deviation[i][3])
            
            xmin.append(minimo[i][0])
            ymin.append(minimo[i][1])
            zmin.append(minimo[i][2])
            rvmin.append(minimo[i][3])
            
            xmax.append(maximo[i][0])
            ymax.append(maximo[i][1])
            zmax.append(maximo[i][2])
            rvmax.append(maximo[i][3])
            
            xrms.append(rms[i][0])
            yrms.append(rms[i][1])
            zrms.append(rms[i][2])
            rvrms.append(rms[i][3])
            
            xsc.append(sign_c[i][0])
            ysc.append(sign_c[i][1])
            zsc.append(sign_c[i][2])
            rvsc.append(sign_c[i][3])
            
            user.append(u)
        
        #janela e Instância
        a = pd.DataFrame(wc)
        
        #medias
        c1 = pd.DataFrame(xmean)
        d1 = pd.DataFrame(ymean)
        e1 = pd.DataFrame(zmean)
        f1 = pd.DataFrame(rvmean)
        
        #desvio padrão
        c2 = pd.DataFrame(xstd)
        d2 = pd.DataFrame(ystd)
        e2 = pd.DataFrame(zstd)
        f2 = pd.DataFrame(rvstd)
        
        #mínimos
        c3 = pd.DataFrame(xmin)
        d3 = pd.DataFrame(ymin)
        e3 = pd.DataFrame(zmin)
        f3 = pd.DataFrame(rvmin)
        
        #máximos
        c4 = pd.DataFrame(xmax)
        d4 = pd.DataFrame(ymax)
        e4 = pd.DataFrame(zmax)
        f4 = pd.DataFrame(rvmax)
        
        #RMS (Root Means Squared)
        c5 = pd.DataFrame(xrms)
        d5 = pd.DataFrame(yrms)
        e5 = pd.DataFrame(zrms)
        f5 = pd.DataFrame(rvrms)
        
        # Mudancas de Sinal
        c6 = pd.DataFrame(xsc)
        d6 = pd.DataFrame(ysc)
        e6 = pd.DataFrame(zsc)
        f6 = pd.DataFrame(rvsc)
        
        # User
        
        df_f = pd.concat([a, 
                          c1, d1, e1, f1,
                          c2, d2, e2, f2,
                          c3, d3, e3, f3,
                          c4, d4, e4, f4,
                          c5, d5, e5, f5,
                          c6, d6, e6, f6,], axis=1, sort=False)
        
        col = ["window",  
               "x_mean", "y_mean", "z_mean", "rv_mean",
               "x_std", "y_std", "z_std", "rv_std",
               "x_min", "y_min", "z_min", "rv_min",
               "x_max", "y_max", "z_max", "rv_max",
               "x_rms", "y_rms", "z_rms", "rv_rms",
               "x_signal_changes", "y_signal_changes", "z_signal_changes", "rv_signal_changes"]
        df_f.columns = col
        # print (df_f)
        df_f = df_f.join(df_hist, how='outer')
                
        b = pd.DataFrame(user, columns = ["user"])
        df_f = df_f.join(b, how='outer')
        
        # print (df_f)
        #usr.nickel = df_f
        
        print('Criando usuário: ', u)
        print('Sessão : ', session)
        print('Tipo Nickel')

        cr_csv.create_csv(u, session, df_f, "user_nickel")  

    return 0

#=============================================================================================

#creating_nickel_files(1, 5, 50)
#creating_nickel_files(2, 5, 50)

'''
session = 1
win = 5
ovlap = 50



sess = create_sess(session)
    
#for u in range(0, len(sess)):
for u in range(0, 1):
    usr = create_user(u, get_user_data(u, session))
    data = usr.data

    print (data)

    c_data = center(usr.data)
    
    rv_data = resultant_vector(c_data)
    c_data["res"] = pd.concat([rv_data], axis=1, sort=False)
    
    # print (c_data)
    usr = create_user(u, c_data)
    
    #w_data = windowed_data(usr, window(win))
    w_data = windowed_data_overlap(usr, window(win), ovlap)
    
    sign_c = signal_change(w_data)
    media = mean_window(w_data) #OK
    standard_deviation = std_window(w_data)
    minimo = min_window(w_data)
    maximo = max_window(w_data)
    rms = root_mean_square(w_data)
    
    # Histogram
    df_hist = histogram_nickel(w_data, 10)
    
    wc = []

    xmean = []
    ymean = []
    zmean = []
    rvmean = []
    
    xstd = []
    ystd = []
    zstd = []
    rvstd = []
    
    xmin = []
    ymin = []
    zmin = []
    rvmin = []
    
    xmax = []
    ymax = []
    zmax = []
    rvmax = []
    
    xrms = []
    yrms = []
    zrms = []
    rvrms = []
    
    xsc = []
    ysc = []
    zsc = []
    rvsc = []
    
    user = []
    
    
    for i in range(0, len(w_data)):
        wc.append(i)

        xmean.append(media[i][0])
        ymean.append(media[i][1])
        zmean.append(media[i][2])
        rvmean.append(media[i][3])
        
        xstd.append(standard_deviation[i][0])
        ystd.append(standard_deviation[i][1])
        zstd.append(standard_deviation[i][2])
        rvstd.append(standard_deviation[i][3])
        
        xmin.append(minimo[i][0])
        ymin.append(minimo[i][1])
        zmin.append(minimo[i][2])
        rvmin.append(minimo[i][3])
        
        xmax.append(maximo[i][0])
        ymax.append(maximo[i][1])
        zmax.append(maximo[i][2])
        rvmax.append(maximo[i][3])
        
        xrms.append(rms[i][0])
        yrms.append(rms[i][1])
        zrms.append(rms[i][2])
        rvrms.append(rms[i][3])
        
        xsc.append(sign_c[i][0])
        ysc.append(sign_c[i][1])
        zsc.append(sign_c[i][2])
        rvsc.append(sign_c[i][3])
        
        user.append(u)
    
    #janela e Instância
    a = pd.DataFrame(wc)
    
    #medias
    c1 = pd.DataFrame(xmean)
    d1 = pd.DataFrame(ymean)
    e1 = pd.DataFrame(zmean)
    f1 = pd.DataFrame(rvmean)
    
    #desvio padrão
    c2 = pd.DataFrame(xstd)
    d2 = pd.DataFrame(ystd)
    e2 = pd.DataFrame(zstd)
    f2 = pd.DataFrame(rvstd)
    
    #mínimos
    c3 = pd.DataFrame(xmin)
    d3 = pd.DataFrame(ymin)
    e3 = pd.DataFrame(zmin)
    f3 = pd.DataFrame(rvmin)
    
    #máximos
    c4 = pd.DataFrame(xmax)
    d4 = pd.DataFrame(ymax)
    e4 = pd.DataFrame(zmax)
    f4 = pd.DataFrame(rvmax)
    
    #RMS (Root Means Squared)
    c5 = pd.DataFrame(xrms)
    d5 = pd.DataFrame(yrms)
    e5 = pd.DataFrame(zrms)
    f5 = pd.DataFrame(rvrms)
    
    # Mudancas de Sinal
    c6 = pd.DataFrame(xsc)
    d6 = pd.DataFrame(ysc)
    e6 = pd.DataFrame(zsc)
    f6 = pd.DataFrame(rvsc)
    
    # User
    
    df_f = pd.concat([a, 
                      c1, d1, e1, f1,
                      c2, d2, e2, f2,
                      c3, d3, e3, f3,
                      c4, d4, e4, f4,
                      c5, d5, e5, f5,
                      c6, d6, e6, f6,], axis=1, sort=False)
    
    col = ["window",  
           "x_mean", "y_mean", "z_mean", "rv_mean",
           "x_std", "y_std", "z_std", "rv_std",
           "x_min", "y_min", "z_min", "rv_min",
           "x_max", "y_max", "z_max", "rv_max",
           "x_rms", "y_rms", "z_rms", "rv_rms",
           "x_signal_changes", "y_signal_changes", "z_signal_changes", "rv_signal_changes"]
    df_f.columns = col
    # print (df_f)
    df_f = df_f.join(df_hist, how='outer')
            
    b = pd.DataFrame(user, columns = ["user"])
    df_f = df_f.join(b, how='outer')
    
    #print (df_f)
    #usr.nickel = df_f
    
    
    create_csv(u, session, df_f, "user_nickel")

        
     '''   
