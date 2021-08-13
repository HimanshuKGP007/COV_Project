import pickle
from helper import draw_embed, create_spectrogram, read_audio, record, save_record, preprocess, get_dataframe, scaler_transform
import os
import librosa
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
#from encoder.inference import plot_embedding_as_heatmap
import sounddevice as sd
import wavio
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 3 10:36:12 2021

@author: DELL
"""
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
features = ['chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth',
       'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4',
       'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11',
       'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18',
       'mfcc19', 'mfcc20']
path = r'C:\Users\DELL\COV_Project\Files\cough-heavy.wav'
#path = 'C:\Kgp\CP\Preparation\COVID Project\Audio Files\Mendeley\data\covid\pos-0421-084-cough-m-50.wav'

def preproces(fn_wav):
    y, sr = librosa.load(fn_wav, mono=True, duration=5)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    feature_row = {        
        'chroma_stft': np.mean(chroma_stft),
        'rmse': np.mean(rmse),
        'spectral_centroid': np.mean(spectral_centroid),
        'spectral_bandwidth': np.mean(spectral_bandwidth),
        'rolloff': np.mean(rolloff),
        'zero_crossing_rate': np.mean(zcr),        
    }
    for i, c in enumerate(mfcc):
        feature_row[f'mfcc{i+1}'] = np.mean(c)
    
    return feature_row

def get_dataframe(feature_row):
    data = pd.DataFrame.from_dict(feature_row, orient='index')
    data2 = data.T
    return data2

feature = preproces(path)
# data = pd.DataFrame.from_dict(feature, orient='index')
# data2 = data.T

def scaler_transform(feature):
    df = pd.read_csv(r"C:\Users\DELL\COV_Project\Files\smote_no_encode.csv")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(np.array(df.iloc[:, :-1]))
    X_s = pd.DataFrame(X, columns = features)
    X_s['label'] = df['label']
    test_normalised = scaler.transform(feature)
    return test_normalised

retro = preproces(path)
retro1 = get_dataframe(retro)
retro2 = scaler_transform(retro1)

import pickle
loaded_model = pickle.load(open(r'C:\Users\DELL\COVID_app\model\finalized_model.sav', 'rb'))

a = loaded_model.predict(retro2)    
    
a[0]






