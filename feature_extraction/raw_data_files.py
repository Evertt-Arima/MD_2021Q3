# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:09:06 2020

@author: Arima

Organiza os dados brutos dos eixos (x, y, z)
por janela e gera um arquivo CSV.

"""

from user import *
from dataImport import *
import numpy as np
import pandas as pd
from statistic import *
import create_csv as cr_csv
import csv
import glob as gb


def data_axis(data):
    """
    Parameters
    ----------
    data : TYPE list
        user data windowed

    Returns
    -------
    x, y, z : TYPE array
        array with the data per each axis
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



def creating_raw_files(session, win, ovlap):

    sess = create_sess(session)
        
    for u in range(0, len(sess)):
        usr = create_user(u, get_user_data(u, session))

        w_data = windowed_data_overlap(usr, window(win), ovlap)
        
        x, y, z = data_axis(w_data)
        
        wc = []
        
        wdw = 0
        for wind in w_data:
            wc.append(wdw)
            wdw += 1
        
        w1 = pd.DataFrame(wc, columns = ["window"])
        x1 = pd.DataFrame(x)
        y1 = pd.DataFrame(y)
        z1 = pd.DataFrame(z)
        
        df_f = pd.concat([w1, x1, y1, z1], axis=1, sort=False)       
        df_f["user"] = u
        
        cr_csv.create_csv(u, session, df_f, "user_raw")
    
    return 0





def creating_raw_rv_files(session, win, ovlap):
    
    sess = create_sess(session)
        
    for u in range(0, len(sess)):
        usr = create_user(u, get_user_data(u, session))
        #data = usr.data
        
        #usr = create_user(u, data)
        
        #w_data = windowed_data(usr, window(win))
        w_data = windowed_data_overlap(usr, window(win), ovlap)
        
        
        x, y, z = data_axis(w_data)
        
        rv_array = []
        
        for i in range(0, len(x)):
            aux = []
            for j in range(0, len(x[i])):
                aux.append([x[i][j], y[i][j], z[i][j]])
            rv_array.append(resultant_vector2(pd.DataFrame(aux)))
        
        df = pd.DataFrame(rv_array)
        
        wc = []
        wdw = 0
        for wind in w_data:
            wc.append(wdw)
            wdw += 1
        w1 = pd.DataFrame(wc, columns = ["window"])
        
        df_f = pd.concat([w1, df], axis=1, sort=False)
        df_f['user'] = u
               

        cr_csv.create_csv(u, session, df_f, "user_raw_resultant")
    return 0



def creating_raw_mean_files(session, win, ovlap):
    
    sess = create_sess(session)
        
    for u in range(0, len(sess)):
        usr = create_user(u, get_user_data(u, session))
        #data = usr.data
        
        #usr = create_user(u, data)
        
        #w_data = windowed_data(usr, window(win))
        w_data = windowed_data_overlap(usr, window(win), ovlap)
        
        
        
        m_data = mean_window(w_data)
        
        wc = []
        xmean = []
        ymean = []
        zmean = []
        user = []
        
        wdw = 0
        
        for i in range(0, len(w_data)):
            wc.append(i)
        
            xmean.append(m_data[i][0])
            ymean.append(m_data[i][1])
            zmean.append(m_data[i][2])
            user.append(u)
            wdw += 1
        
        w1 = pd.DataFrame(wc)
        x1 = pd.DataFrame(xmean)
        y1 = pd.DataFrame(ymean)
        z1 = pd.DataFrame(zmean)
        u1 = pd.DataFrame(user)
        
        df_f = pd.concat([w1, x1, y1, z1, u1], axis=1, sort=False)
        df_f.columns = ["window", "x_mean", "y_mean", "z_mean", "user"] 
        
        
        cr_csv.create_csv(u, session, df_f, "user_raw_mean")


    return 0


#=====================================================================================

'''
creating_raw_files(1, 5, 50)
creating_raw_files(2, 5, 50)


creating_raw_rv_files(1, 5, 50)
creating_raw_rv_files(2, 5, 50)


creating_raw_mean_files(1, 5, 50)
creating_raw_mean_files(2, 5, 50)

'''
