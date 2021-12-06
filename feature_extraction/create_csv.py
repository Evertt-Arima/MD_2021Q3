# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:11:41 2020

@author: Arima
"""


def create_csv(u, session, df_f, feat):
    
    if feat=="user_raw":
        if u < 10:
            if session == 1:
                name = "features/user_raw/session1/user0" + str(u) + ".csv"
            else:
                name = "features/user_raw/session2/user0" + str(u) + ".csv"
        else:
        
            if session == 1:
                name = "features/user_raw/session1/user" + str(u) + ".csv"
            else:
                name = "features/user_raw/session2/user" + str(u) + ".csv"
        
    elif feat == "user_raw_resultant":
        if u < 10:
            if session == 1:
                name = "features/user_raw_resultant/session1/user0" + str(u) + ".csv"
            else:
                name = "features/user_raw_resultant/session2/user0" + str(u) + ".csv"
        else:
            if session == 1:
                name = "features/user_raw_resultant/session1/user" + str(u) + ".csv"
            else:
                name = "features/user_raw_resultant/session2/user" + str(u) + ".csv"
    
    elif feat == "user_raw_mean":
        
        if u < 10:
            if session == 1:
                name = "features/user_raw_mean/session1/user0" + str(u) + ".csv"
            else:
                name = "features/user_raw_mean/session2/user0" + str(u) + ".csv"
        else:
        
            if session == 1:
                name = "features/user_raw_mean/session1/user" + str(u) + ".csv"
            else:
                name = "features/user_raw_mean/session2/user" + str(u) + ".csv"

    elif feat == "FFT_features":
        
        if u < 10:
            if session == 1:
                name = "features/FFT5_features/session1/user0" + str(u) + ".csv"
            else:
                name = "features/FFT5_features/session2/user0" + str(u) + ".csv"
        else:
            if session == 1:
                name = "features/FFT5_features/session1/user" + str(u) + ".csv"
            else:
                name = "features/FFT5_features/session2/user" + str(u) + ".csv"
        
        
    elif feat == "FFT_full_features":
        if u < 10:
            if session == 1:
                name = "features/FFT_full_features/session1/user0" + str(u) + ".csv"
            else:
                name = "features/FFT_full_features/session2/user0" + str(u) + ".csv"
        else:
            if session == 1:
                name = "features/FFT_full_features/session1/user" + str(u) + ".csv"
            else:
                name = "features/FFT_full_features/session2/user" + str(u) + ".csv"
        
        
    elif feat == "user_nickel":
        if u < 10:
            if session == 1:
                name = "features/user_nickel/session1/user0" + str(u) + ".csv"
            else:
                name = "features/user_nickel/session2/user0" + str(u) + ".csv"
        else:
        
            if session == 1:
                name = "features/user_nickel/session1/user" + str(u) + ".csv"
            else:
                name = "features/user_nickel/session2/user" + str(u) + ".csv"
        
        
    elif feat == "user_kwapisz":
        if u < 10:
            if session == 1:
                name = "features/user_kwapisz/session1/user0" + str(u) + ".csv"
            else:
                name = "features/user_kwapisz/session2/user0" + str(u) + ".csv"
        else:
        
            if session == 1:
                name = "features/user_kwapisz/session1/user" + str(u) + ".csv"
            else:
                name = "features/user_kwapisz/session2/user" + str(u) + ".csv"
        
        
    df_f.to_csv(name, index=False)