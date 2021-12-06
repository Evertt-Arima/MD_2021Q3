# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 03:13:30 2020


@author: Arima

Este arquivo processa os dados
segundo os parâmetros do
trabalho de Kwapisz
"""


from user import *
from dataImport import *
import create_csv as cr_csv
import numpy as np
import pandas as pd
from statistic import *
import csv
import glob as gb


def average_absolute_diff(data):
    '''
    Parameters
    ----------
    data : TYPE list
        list with windowed data.

    Returns
    -------
    aad : TYPE list
        list with mean of Absolute difference between
        the axis value minus the mean of the axis within the window.
    '''
    media = mean_window(data) 
    aad_aux = []
    for i in range(0, len(data)):
        aux = []
        for j in range(0, len(data[i])):
            vals = []
            for k in range(0, len(data[i][j])):
                num = data[i][j][k] - media[i][k]
                vals.append(num)
            aux.append(vals)
        aad_aux.append(aux)
    m_aad = mean_window(aad_aux)
    aad = []
    for line in m_aad:
        aad.append(list(map(abs, line)))
    return aad



def average_resultant_accel(data, win, ovlap):
    '''
    Parameters
    ----------
    data : TYPE DataFrame
        data NOT windowed, with all 3 axis.
    win : TYPE int
        time in seconds of the window.

    Returns
    -------
    ara : TYPE list
        Sum of Magnetude Vectors of the window over lenght of the window.

    '''
    w_len = window(win)
    rv_data = resultant_vector(data)
    ara_aux = windowed_data2(rv_data, w_len, ovlap)
    ara = []
    for i in range(len(ara_aux)):
        soma = 0
        for j in range(len(ara_aux[i])):
            soma += ara_aux[i][j][0]
        v_ara = soma / len(ara_aux[i])
        ara.append(v_ara)
    return ara

def histogram_kwpisz(data, box):
    '''
    Parameters
    ----------
    data : TYPE list
        DESCRIPTION
        windowed data (x, y, z).
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
        aux = []
        for j in range(len(data[i])):
            x.append(data[i][j][0])
            y.append(data[i][j][1])
            z.append(data[i][j][2])
        aux_x = np.asarray(np.histogram(x, bins=box))
        aux_x = aux_x[0]
        aux_y = np.asarray(np.histogram(y, bins=box))
        aux_y = aux_y[0]
        aux_z = np.asarray(np.histogram(z, bins=box))
        aux_z = aux_z[0]
        aux = np.append(aux_x, [aux_y, aux_z])
        hist.append(aux)
    df_hist = pd.DataFrame(hist)
    return df_hist


def creating_kwapisz_files(session, win, ovlap):

    sess = create_sess(session)
        
    for u in range(0, len(sess)):

        usr = create_user(u, get_user_data(u, session))
        data = usr.data
        
        #w_data = windowed_data(usr, window(win))
        w_data = windowed_data_overlap(usr, window(win), ovlap)
        
        
        # Mean per Axis per Window
        media = mean_window(w_data) 
        
        # Standard Deviation per Axis per Window
        st_dev = std_window(w_data)
        
        # Average Absolute Difference
        aad = average_absolute_diff(w_data)
        # print (aad)
        
        # Average Resultant Acceleration
        ara = average_resultant_accel(data, win, ovlap)
        
        # Time Between Peaks
        tbp_x, tbp_y, tbp_z = time_btw_peaks(w_data) #list Windowed Data
        # print (tbp_x)
        # print (tbp_y)
        # print (tbp_z)
        
        # Histogram
        df_hist = histogram_kwpisz(w_data, 10)
        
        wc = []

        xmean = []
        ymean = []
        zmean = []
        
        xstd = []
        ystd = []
        zstd = []
        
        xaad = []
        yaad = []
        zaad = []
        
        user = []
        
        
        for i in range(0, len(w_data)):
            wc.append(i)

            xmean.append(media[i][0])
            ymean.append(media[i][1])
            zmean.append(media[i][2])
            
            xstd.append(st_dev[i][0])
            ystd.append(st_dev[i][1])
            zstd.append(st_dev[i][2])

            xaad.append(aad[i][0])
            yaad.append(aad[i][1])
            zaad.append(aad[i][2])

            user.append(u)
        
        #janela e Instância
        df_wdw = pd.DataFrame(wc)
        
        #medias
        df_mean_x = pd.DataFrame(xmean)
        df_mean_y = pd.DataFrame(ymean)
        df_mean_z = pd.DataFrame(zmean)
        
        #desvio padrão
        df_std_x = pd.DataFrame(xstd)
        df_std_y = pd.DataFrame(ystd)
        df_std_z = pd.DataFrame(zstd)
        
        # Average Absolute Differences
        df_aad_x = pd.DataFrame(xaad)
        df_aad_y = pd.DataFrame(xaad)
        df_aad_z = pd.DataFrame(xaad)

        # Average Resultant Accelerations
        df_ara = pd.DataFrame(ara)
        
        # Time between peaks within the window
        df_tpx = pd.DataFrame(tbp_x)
        df_tpy = pd.DataFrame(tbp_y)
        df_tpz = pd.DataFrame(tbp_z)
        
        
        df_f = pd.concat([df_wdw, 
                          df_mean_x, df_mean_y, df_mean_z,
                          df_std_x, df_std_y, df_std_z,
                          df_aad_x, df_aad_y, df_aad_z,
                          df_ara,
                          df_tpx, df_tpy, df_tpz
                          ], axis=1, sort=False)
        
        col = ["window",  
               "x_mean", "y_mean", "z_mean", 
               "x_std", "y_std", "z_std", 
               "x_abs_diff", "y_abs_diff", "z_abs_diff", 
               "mean_res_accel", 
               "x_tm_peaks", "y_tm_peaks", "z_tm_peaks"
               ]
        df_f.columns = col
        # print(df_f)
        
        df_f = df_f.join(df_hist, how='outer')
        
        
        b = pd.DataFrame(user, columns = ["user"])
        df_f = df_f.join(b, how='outer')
        # print (df_f)
        
        #usr.kwapisz = df_f
        

        cr_csv.create_csv(u, session, df_f, "user_kwapisz")

    return 0


#=================================================================================

#creating_kwapisz_files(1, 5, 50)
#creating_kwapisz_files(2, 5, 50)

'''







u = 0
session = 1
win = 3


sess = gb.glob("user_coordinates/*session1*.txt")

usr = create_user(u, get_user_data(u, session))
data = usr.data

w_data = windowed_data(usr, window(win))

# Histogram
df_hist = histogram_kwpisz(w_data, 10)
# print (df_hist)


# Mean per Axis per Window
media = mean_window(w_data) 

# Standard Deviation per Axis per Window
st_dev = std_window(w_data)

# Average Absolute Difference
aad = average_absolute_diff(w_data)
# print (aad)

# Average Resultant Acceleration
ara = average_resultant_accel(data, win)

# Time Between Peaks
tbp_x, tbp_y, tbp_z = time_btw_peaks(w_data) #list Windowed Data
# print (tbp_x)
# print (tbp_y)
# print (tbp_z)



wc = []

xmean = []
ymean = []
zmean = []

xstd = []
ystd = []
zstd = []

xaad = []
yaad = []
zaad = []

user = []


for i in range(0, len(w_data)):
    wc.append(i)

    xmean.append(media[i][0])
    ymean.append(media[i][1])
    zmean.append(media[i][2])
    
    xstd.append(st_dev[i][0])
    ystd.append(st_dev[i][1])
    zstd.append(st_dev[i][2])

    xaad.append(aad[i][0])
    yaad.append(aad[i][1])
    zaad.append(aad[i][2])

    user.append(u)

#janela e Instância
df_wdw = pd.DataFrame(wc)

#medias
df_mean_x = pd.DataFrame(xmean)
df_mean_y = pd.DataFrame(ymean)
df_mean_z = pd.DataFrame(zmean)

#desvio padrão
df_std_x = pd.DataFrame(xstd)
df_std_y = pd.DataFrame(ystd)
df_std_z = pd.DataFrame(zstd)

# Average Absolute Differences
df_aad_x = pd.DataFrame(xaad)
df_aad_y = pd.DataFrame(yaad)
df_aad_z = pd.DataFrame(zaad)

# Average Resultant Accelerations
df_ara = pd.DataFrame(ara)

# Time between peaks within the window
df_tpx = pd.DataFrame(tbp_x)
df_tpy = pd.DataFrame(tbp_y)
df_tpz = pd.DataFrame(tbp_z)



'''