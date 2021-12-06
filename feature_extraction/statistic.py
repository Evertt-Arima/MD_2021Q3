# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 22:03:54 2020

@author: Arima

Este arquivo contém todos os métodos de extração
estatísticas que serão utilizadas no projeto, 
considerando as abordagens adotadas.

"""

from user import *
from dataImport import *
import numpy as np
import pandas as pd



def mean_window(data):
    """
    Parameters
    ----------
    data : TYPE list
        user data windowed
        
    Returns
    -------
    m : TYPE list
    """
    m = []
    for i in range(0, len(data)):
        aux = np.mean(data[i], axis = 0).tolist()
        m.append(aux)
    return m


def columns_means (df):
    """
    Parameters
    ----------
    data : TYPE dataframe
        user data NOT windowed
        
    Returns
    -------
    m : TYPE list with means of input DataFrame
    """
    df_mean = df.mean(axis=0).tolist()
    return df_mean



def mega_mean():
    """
    Returns
    -------
    mm : TYPE Series
        Series with means (x, y, z, resultant vector) from all genuine user.
    """

    sess = gb.glob("data/Original Data/user_coordinates/*session1*.txt")
    
    cols = ["mega_x_mean", "mega_y_mean", "mega_z_mean"]
    col_means = []
    
    
    for u in range(0, len(sess)): 
        
        usr = create_user(u, get_user_data(u, 1))
        data = usr.data
        
        # rv_data = pd.DataFrame(resultant_vector(data))
        # data["res"] = pd.concat([rv_data], axis=1, sort=False)
    
        col_means.append(columns_means(data))
        
    
    c = pd.DataFrame(col_means)
    c.columns = cols
    
    mm = columns_means(c)
    
    return mm



def center(df):
    """
    Parameters
    ----------
    data : TYPE dataframe
        user data NOT windowed from genuine user
    Returns
    -------
    m : TYPE dataframe centered (value - means)
    """
    df_mean = mega_mean()
    df1 = []
    for _ in df:
        df1.append(df - df_mean)
    dfx = pd.DataFrame(df1[0])
    return dfx


# def center_imp(user, df):
#     """
#     Parameters
#     ----------
#     data : TYPE dataframe
#         user data NOT windowed from impostor user

#     Returns
#     -------
#     m : TYPE dataframe centered (value - means)
#     """
#     df_mean = mega_mean()
    
#     aux = df_mean.loc[user, :].tolist()
#     # print(aux)
#     dfx = df - aux
#     # print (dfx)
#     return dfx




def min_window(data):
    """
    Parameters
    ----------
    data : TYPE dataframe
        user data windowed
    Returns
    -------
    mini : TYPE list 
    """
    mini = []
    for i in range(0, len(data)):
        aux = np.min(data[i], axis = 0)
        mini.append(aux.tolist())
    return mini



def max_window(data):
    """
    Parameters
    ----------
    data : TYPE dataframe
        user data windowed
    Returns
    -------
    maxi : TYPE list 
    """
    maxi = []
    for i in range(0, len(data)):
        aux = np.max(data[i], axis = 0).tolist()
        maxi.append(aux)
    return maxi



def std_window(data):
    """
    Parameters
    ----------
    data : TYPE dataframe
        user data windowed
    Returns
    -------
    st_dev : TYPE list 
    """
    st_dev = []
    for i in range(0, len(data)):
        aux = np.std(data[i], axis = 0).tolist()
        st_dev.append(aux)
#    print (st_dev[-1])
    return st_dev



def resultant_vector(data):
    """
    Parameters
    ----------
    vetor : TYPE dataframe
        user data NOT windowed
    
    Returns
    -------
    res_vector : TYPE dataframe
        Magnetude vector of axis x + y + z, per instance (row) 
        return a Pandas DataFrame of same lenght of the input parameter
        
    Note:
    -----
        Also refered as Magnetude Vector in academic articles.
    """
    aux = data.values
    magn = []
    for line in aux:
        m = ((line[0])**2 + (line[1])**2 + (line[2])**2)**(1/2)
        magn.append(m)
    res_vector = pd.DataFrame(magn)
    return res_vector



def resultant_vector2(data):
    """
    Parameters
    ----------
    vetor : TYPE dataframe
        user data NOT windowed
    
    Returns
    -------
    res_vector : TYPE list
        Magnetude vector from axis x + y + z, per instance (row) 
        return a list of same lenght of the input parameter
        
    Note:
    -----
        Also refered as Magnetude Vector in academic articles.
        Magnetude Vector = Root(a^2 + b^2 + c^2)
    """
    aux = data.values
    magn = []
    for line in aux:
        m = ((line[0])**2 + (line[1])**2 + (line[2])**2)**(1/2)
        magn.append(m)
    return magn



def root_mean_square(data):
    """
    Parameters
    ----------
    data : TYPE list
        user data windowed
    Returns
    -------
    rms : TYPE list 
        list with the Root Mean Squares within the window
    """
    rms = []
    for window in data:
        aux = [0, 0, 0, 0]
        n = len(window)
        for instance in window:
            aux[0] += (instance[0])**2
            aux[1] += (instance[1])**2
            aux[2] += (instance[2])**2
            aux[3] += (instance[3])**2
        aux[0] = ((aux[0])/n)**1/2
        aux[1] = ((aux[1])/n)**1/2
        aux[2] = ((aux[2])/n)**1/2
        aux[3] = ((aux[3])/n)**1/2
        rms.append(aux)   
    return rms



def signal_change(data):
    '''
    Parameters
    ----------
    data : TYPE list with Resultant Vector
        list windowed
    Returns
    -------
    change : TYPE list
        Returns a list with sum of signal changes within each window.
    '''
    change = []
    for index in data:
        sign_x = None
        sign_y = None
        sign_z = None
        sign_rv = None
        cont_x = 0
        cont_y = 0
        cont_z = 0
        cont_rv = 0
        for row in index:
            #print (row)
#            aux_change = []
            if row == index[0]:
                for i in range(0, len(row)):
                    if i == 0:
                        if row[i] > 0:
                            sign_x = True
                        else:
                            sign_x = False
                    elif i == 1:
                        if row[i] > 0:
                            sign_y = True
                        else:
                            sign_y = False
                    elif i == 2:
                        if row[i] > 0:
                            sign_z = True
                        else:
                            sign_z = False
                    else:
                        if row[i] > 0:
                            sign_rv = True
                        else:
                            sign_rv = False
            else:
                for i in range(0, len(row)):
                    if i == 0:
                        if row[i] > 0:
                            aux_sign_x = True
                        else:
                            aux_sign_x = False
                        if sign_x != aux_sign_x:
                            sign_x = aux_sign_x
                            cont_x += 1
                    elif i == 1:
                        if row[i] > 0:
                            aux_sign_y = True
                        else:
                            aux_sign_y = False
                        if sign_y != aux_sign_y:
                            sign_y = aux_sign_y
                            cont_y += 1
                    elif i == 2:
                        if row[i] > 0:
                            aux_sign_z = True
                        else:
                            aux_sign_z = False
                        if sign_z != aux_sign_z:
                            sign_z = aux_sign_z
                            cont_z += 1
                    else:
                        if row[i] > 0:
                            aux_sign_rv = True
                        else:
                            aux_sign_rv = False
                        if sign_rv != aux_sign_rv:
                            sign_rv = aux_sign_rv
                            cont_rv += 1
        change.append([cont_x, cont_y, cont_z, cont_rv])
    return change



def dist_btw_peaks(data):
    """
    Parameters
    ----------
    a : TYPE list Windowed Data
        data of axis x, y, z.
    Returns
    -------
    dist : TYPE array
        Distance between peaks in each window.
        The array of same lenght of windows quantity.
    """
    dist_x = []
    dist_y = []
    dist_z = []   
    
    for w in data: 
        px1 = 0
        px2 = 1
        py1 = 0
        py2 = 1
        pz1 = 0
        pz2 = 1

        for i in range (2, len(w)):
            if (w[i][0] > w[px1][0]) or (w[i][0] > w[px2][0]):
                if (w[px2][0] > w[px1][0]):
                    px1 = i
                else:
                    px2 = i
            if (w[i][1] > w[py1][1]) or (w[i][1] > w[py2][1]):
                if (w[py2][0] > w[py1][0]):
                    py1 = i
                else:
                    py2 = i 
            if (w[i][2] > w[pz1][2]) or (w[i][2] > w[pz2][2]):
                if (w[pz2][0] > w[pz1][0]):
                    pz1 = i
                else:
                    pz2 = i
        # print(px1, px2, py1, py2, pz1, pz2)
        if px1 < px2:
            d = px2 - px1
        else:
            d = px1 - px2
        dist_x.append(d)
        if py1 < py2:
            d = py2 - py1
        else:
            d = py1 - py2
        dist_y.append(d)
        if pz1 < pz2:
            d = pz2 - pz1
        else:
            d = pz1 - pz2
        dist_z.append(d)
        
    return dist_x, dist_y, dist_z


def time_btw_peaks(data):
    """
    Parameters
    ----------
    data : TYPE list Windowed Data
        distance in instances of peaks
    Returns
    -------
    peak_time : TYPE array
        Time (in milisseconds) between peaks in each window.
        The array of same lenght of windows quantity.
    """
    dx, dy, dz = dist_btw_peaks(data)
    pt_x = []
    pt_y = []
    pt_z = []

    for i in range(0, len(dx)):
        t1 = (dx[i] * 1000) / 40
        pt_x.append(t1)
    for line in dy:
        t2 = (line * 1000) / 40
        pt_y.append(t2)
    for line in dz:
        t3 = (line * 1000) / 40
        pt_z.append(t3)
        
    return pt_x, pt_y, pt_z



'''
dados = get_user_data(0, 1)
# rv_data = pd.DataFrame(resultant_vector(dados))
# dados["res"] = pd.concat([rv_data], axis=1, sort=False)
usr = create_user(0, dados)
#print (usr.data[0:3][0:1])
#print (len(usr.data[1:]))
#print ("\n")
wdw = window(3)
#print('janela = ', wdw, '\n')
#print (len(usr.data) / wdw)
cut_size = int(len(usr.data) / wdw)
#print('cut = ', cut_size)
#print (usr.data)
teste = windowed_data(usr, wdw)
#print(teste)




print('=======================TESTE===============\n')
# print(teste[0:2])
#print(teste[-2])
#print(teste)
#print(teste[0][0])
#print(len(teste[0]))
#print(usr.data[0])
print("\n==================CENTER - Genuine=====================")
centro_g = center_gen(usr.data)
print (len(centro_g))
print (centro_g)
print("\n==================CENTER - Impostor=====================")
centro_i = center_imp(usr.user, usr.data)
print (len(centro_i))
print (centro_i)
print("\n==================MÉDIA=====================")
media = mean_window(teste)
print (len(media))
print (media)
print("\n==================MÍNIMO=====================")
minimo = min_window(teste)
print (len(minimo))
print (minimo)
print("\n==================MÁXIMO=====================")
maximo = max_window(teste)
print (len(maximo))
print (maximo)



print("\n==================DESVIO PADRÃO=====================")
standard_deviation = std_window(teste)
print (len(standard_deviation))
#print (standard_deviation)



print("\n==================SOMA DOS VETORES=====================")
res_vect = resultant_vector(usr.data)
print (len(res_vect))
print (res_vect)

print("\n==================ROOT MEAN SQUARE=====================")
# rms = root_mean_square(teste)
# print (len(rms))
# print (rms)

print("\n==================MAGNETUDE VECTOR=====================")
mag = magnetude_vector(teste)
print (mag)

print("\n==================DISTANCE BETWEEN PEAKS=====================")
d_peaks = dist_btw_peaks(teste)
print (d_peaks)

print("\n==================TIME BETWEEN PEAKS=====================")
t_peaks = time_btw_peaks(teste)
print (t_peaks)

print("\n=============CHANGE SIGN PER WINDOW===============")
c = signal_change(teste)
print (c)   
 
print('==================================================')

print('\nnumpy teste\n')
u = np.array(teste)
print (u[0])
print (u[0][0])
print (u[0][0][0])
print('\n')
'''















