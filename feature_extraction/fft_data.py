#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 00:17:39 2020

@author: Arima

Este arquivo gera o FFT (Fast Fourier Transform)
por usuário e cria um arquivo CSV baseado nos 5 primeiros
resultados da FFT
"""

from dataImport import *
import create_csv as cr_csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
import csv



def data_feat_fft(data):
    """
    Parameters
    ----------
    data : TYPE DataFrame
        user data windowed
    interest : TYPE int
        amount of instances which FFT should consider per window

    Returns
    -------
    x, y, z : TYPE array
        array with the interest amount in each axis
    """
    x = []
    y = []
    z = []
    # Loop with window to extract features
    for window in range (0, len(data)):
        x1 = []
        y1 = []
        z1 = []
        for i in range (0, len(data[window])):
            x1.append(data[window][i][0])
            y1.append(data[window][i][1])
            z1.append(data[window][i][2])
        x.append(x1)
        y.append(y1)
        z.append(z1)
    return x, y, z


def data_fft(data):
    """
    Parameters
    ----------
    data : TYPE array
        axis array of interest instances
    Returns
    -------
    fft_array : TYPE list
    """
    n = len(data)
    Lx = n
    
    x = np.linspace(0, Lx, n)
    y = data
    
    freqs = fftfreq(n)
    mask = freqs > 0
    
    fft_vals = fft(y)
    #print (fft_vals)
    fft_theo = 2.0 * np.abs(fft_vals/n)
#    print (fft_theo)
    
#    plt.figure(1)
#    plt.title("Original Signal")
#    plt.plot(x,y) # label = "original")
#    #plt.legend()
#    
#    
#    plt.figure(2)
#    plt.plot(freqs, fft_vals, label = "raw fft values")
#    plt.title("Raw FFT values")
#    
#    plt.figure(3)
#    plt.plot(freqs[mask], fft_theo[mask], label = "true fft values")
#    plt.title("True FFT values")
#    plt.plot()
    
    fft_array = []
    for wdw in range (0, len(data)):
        aux = list(abs(np.fft.fft(data[wdw])))
        fft_array.append(aux)
    return fft_array


def creating_fft_files(session, win, ovlap):
    # sess1_files = gb.glob("user_coordinates/*session1*.txt")
    # sess2_files = gb.glob("user_coordinates/*session2*.txt")
    sess = create_sess(session)
    
    
    for i in range(0, len(sess)):
        usr = create_user(i, get_user_data(i, session))
        wdw = window(win)
        
        
        #w_data = windowed_data(usr, window(win))
        data_wdw = windowed_data_overlap(usr, wdw, ovlap)
        #data_wdw = windowed_data(usr, wdw)
        
        x, y, z = data_feat_fft(data_wdw)
#        print (x)
#        print (y)
#        print (z)
        z_fft = data_fft(z)
        x_fft = data_fft(x)
        y_fft = data_fft(y)
        
        w_col = ["window"] 
        x_col = ["x1", "x2", "x3", "x4", "x5"]
        y_col = ["y1", "y2", "y3", "y4", "y5"]
        z_col = ["z1", "z2", "z3", "z4", "z5"]
        
        wa = []
        xa = []
        ya = []
        za = []
        for k in range(0, len(x_fft)):
            xb = []
            yb = []
            zb = []
            for j in range(0, 5):
                xb.append(x_fft[k][j])
                yb.append(y_fft[k][j])
                zb.append(z_fft[k][j])
            wa.append(k)
            xa.append(xb)
            ya.append(yb)
            za.append(zb)
        
        df_w = pd.DataFrame(wa, columns = w_col)
        df_x = pd.DataFrame(xa, columns = x_col)
        df_y = pd.DataFrame(ya, columns = y_col)
        df_z = pd.DataFrame(za, columns = z_col)
        df = pd.concat([df_w, df_x, df_y, df_z], axis=1, sort =False)
        df["user"] = usr.user
        
        # print (df)
        
        cr_csv.create_csv(usr.user, session, df, "FFT_features")

        
    return 0



def creating_fft_full_files(session, win, ovlap):

    sess = create_sess(session)
        
    for u in range(0, len(sess)):
        usr = create_user(u, get_user_data(u, session))
        wdw = window(win)
        
        
        #w_data = windowed_data(usr, window(win))
        data_wdw = windowed_data_overlap(usr, wdw, ovlap)
        #data_wdw = windowed_data(usr, wdw)
        
        x, y, z = data_feat_fft(data_wdw)
        
        z_fft = data_fft(z)
        x_fft = data_fft(x)
        y_fft = data_fft(y)
        
        wa = []
        xa = []
        ya = []
        za = []
        for k in range(0, len(x_fft)):
            xb = []
            yb = []
            zb = []
            for j in range(0, len(x_fft[k])):
                xb.append(x_fft[k][j])
                yb.append(y_fft[k][j])
                zb.append(z_fft[k][j])
            wa.append(k)
            xa.append(xb)
            ya.append(yb)
            za.append(zb)
        
        df_w = pd.DataFrame(wa, columns = ["window"])
        df_x = pd.DataFrame(xa)
        df_y = pd.DataFrame(ya)
        df_z = pd.DataFrame(za)
        df = pd.concat([df_w, df_x, df_y, df_z], axis=1, sort =False)
        df["user"] = usr.user
        
        # print (df)
        cr_csv.create_csv(usr.user, session, df, "FFT_full_features")
        
        
    return 0


#==================================================================================

'''

creating_fft_files(1, 5, 50)
creating_fft_files(2, 5, 50)


creating_fft_full_files(1, 5, 50)
creating_fft_full_files(2, 5, 50)

'''