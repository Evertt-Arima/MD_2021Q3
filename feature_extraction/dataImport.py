# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:12:21 2020

@author: Arima

Este arquivo lê arquivos TXT ou CSV
o pode serparar os eixos e/ou 
janelar os dados mediante mediante um determinado
tempo em segundos.
É preciso a importação de user.py
"""
import glob as gb
import numpy as np
import pandas as pd
#from csv import reader
from user import User

def create_sess(session):
    if session==1:
        sess = gb.glob("data/Original Data/user_coordinates/*session1*.txt")
    else:
        sess = gb.glob("data/Original Data/user_coordinates/*session2*.txt")
    return sess


def get_user_data(user, session):
    """
    Get the data file of the user and session
    passed as parameters, and
    returns the data as a list (array)
    os lists.
    """
    if session == 1:
        path_session = gb.glob("data/Original Data/user_coordinates/*session1*.txt")
    elif session == 2:
        path_session = gb.glob("data/Original Data/user_coordinates/*session2*.txt")
    file = path_session[user]
    #opened_file = open(file)
    #read_file = reader(opened_file)
    #data = list(read_file)
    data = pd.read_csv(file, sep=",", header=None)
    #data.columns = ["x", "y", "z"]    
    return data


def separate_axis(data):
    """
    Separate the data passed in 3 different
    arrays, according to the axis.
    """
    a = data[0]
    b = data[1]
    c = data[2]
    
    return a, b, c


def create_user(user, data):
    """
    Create an User() object.
    """
    usr = User(user, data)
    return usr


def window(seconds):
    """
    Separate the data in windows of
    the time passed, in seconds,
    considering a frequency of
    40 Hz.
    """
    wdw = int(seconds * 40)
    return wdw



def windowed_data(user, window):
    """
    Separate the data of a user in windows
    and return a list of numpy arrays where each index
    contains the data of that specific window.
    """
    data = user.data
    #print('data lenght = ',len(data))
    cut_size = int(len(data) / window)
    #print ('cut = ', cut_size)
    start = 0
    end = window
    windowed = []
    size = len(data)
    for i in range(cut_size*2):
        for rows in range(start, end, window):
            #print(f'start: {start}')
            #print(f'end: {end}')
            aux = []
            if end <= size:
                aux = data[start:end]
            #print('=============AUX==========\n', aux)
            a = np.array(aux).tolist()
        windowed.append(a)
        start += int(window/2)
        end += int(window/2)
    return windowed

def windowed_data2(data, window, overlap):
    """
    Separate the data of a user in windows
    and return a list of numpy arrays where each index
    contains the data of that specific window.
    """
    ovlap = (100 - overlap)/100
    #print('data lenght = ',len(data))
    cut_size = int(len(data) / window)
    #print ('cut = ', cut_size)
    start = 0
    end = window
    windowed = []
    size = len(data)
    for i in range(int(cut_size/ovlap)):
        for rows in range(start, end, window):
            #print(f'start: {start}')
            #print(f'end: {end}')
            aux = []
            if end <= size:
                aux = data[start:end]
            #print('=============AUX==========\n', aux)
            a = np.array(aux).tolist()
        windowed.append(a)
        start += int(window/2)
        end += int(window/2)
    return windowed


def windowed_data_overlap(user, window, overlap):
    """
    Separate the data of a user in windows
    with specific percentage of overlap
    and return a list of numpy arrays where each index
    contains the data of that specific window.
    
    """
    
    data = user.data
    ovlap = (100 - overlap)/100
    
    #print('data lenght = ',len(data))
    cut_size = int(len(data) / window)
    #print ('cut = ', cut_size)
    start = 0
    end = window
    windowed = []
    size = len(data)
    while end < size:
        for rows in range(start, end, window):
            #print(f'start: {start}')
            #print(f'end: {end}')
            aux = []
            if end <= size:
                aux = data[start:end]
            #print('=============AUX==========\n', aux)
            a = np.array(aux).tolist()
        windowed.append(a)
        start += int(window * ovlap)
        end += int(window * ovlap)
    return windowed

'''
# These are tests and should be ignored

dados = get_user_data(0, 1)
usr = create_user(0, dados)

print (len(usr.data))
print ("\n")
wdw = window(10)
print ('cut 1 = ', len(usr.data) / wdw)
print ("\n")
#cut_size = int(len(usr.data) / wdw)

#print(cut_size)


teste = windowed_data(usr, wdw)
teste2 = windowed_data2(usr.data, wdw)

#print(len(teste))
#print('===========TESTE============\n', teste)
print(len(teste[0]))
print(teste[-1])
print ("\n")
#print(usr.data)

print(np.mean(teste[0], axis = 0))

'''
