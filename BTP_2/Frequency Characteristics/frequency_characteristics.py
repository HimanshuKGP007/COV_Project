#from scipy import fft
from scipy import signal as sig
from scipy.fft import fft, fftfreq
from IPython.display import Audio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import wave
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import librosa, librosa.display
import IPython.display as ipd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import warnings
warnings.simplefilter('ignore')

# Testing Files present at Files\cough_wavs_sample

# https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files --> Answer 2
def fft_new(file):
    fs_rate, signal = wavfile.read(file)
    #print ("Frequency sampling", fs_rate)
    l_audio = len(signal.shape)
    #print ("Channels", l_audio)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    #N = signal.shape[0]
    N = 20000
    #print ("Complete Samplings N", N)
    secs = N / float(fs_rate)
    #print ("secs", secs)
    Ts = 1.0/fs_rate # sampling interval in time
    #print ("Timestep between samples Ts", Ts)
    t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
    FFT = abs(fft(signal))
    FFT_side = FFT[range(N//2)] # one side FFT range
    freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
    fft_freqs = np.array(freqs)
    freqs_side = freqs[range(N//2)] # one side frequency range
    fft_freqs_side = np.array(freqs_side)
#     p = plt.plot(freqs_side, abs(FFT_side), "b") # plotting the positive fft spectrum
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Magnitude')
#     plt.show()
    mag = abs(FFT_side)
    freq = freqs_side
    return freq, mag

#file = r'C:\Users\DELL\COV_Project\Files\cough_wavs_sample\cleaned_data\Positive\1192_Positive_male_28.wav'
#freq, mag = fft_new(file)

def fft_plot(file):
    #file = '../input/covid-cough-wavs/cleaned_data/Positive/1192_Positive_male_28.wav'
    signal, sr = librosa.load(file)
    X = np.fft.fft(signal)
    X_mag = np.absolute(X)
    
    plt.figure(figsize=(18, 5))
    
    f = np.linspace(0, sr, len(X_mag))
    f_ratio = 0.05
    f_bins = int(len(X_mag)*f_ratio)  
    
    plt.plot(f[:f_bins], X_mag[:f_bins])
    plt.xlabel('Frequency (Hz)')
    source = '../input/covid-cough-wavs/cleaned_data/Positive'
    if source in file:
        name = file.replace(source + '/','')
    plt.title(name)
    return f[:f_bins], X_mag[:f_bins]


def createSeries (series_list):
    series_list = pd.Series(series_list)  
    return series_list

def freq_bin_vs_amp(frequency_list, amplitude_list):
    frequencies = createSeries(frequency_list) 
    Amplitude = createSeries(amplitude_list)
    
    # create a dictonary
    data = {"frequencies": frequencies,
            "Amplitude": Amplitude}
    df = pd.concat(data, axis = 1)
    #Create Bins
    squares = []
    # for i in range(11):
    #     squares.append(i * 100)
    for i in range(51):
        squares.append(i * 20)
    bins = squares
    df['binned_freq'] = pd.cut(df['frequencies'], bins)
    # Extract Maximum amplitude for that Bin
    # ---------------------------------------------------------------------------------------------
    # PROBLEM
    # ---------------------------------------------------------------------------------------------
    df = df[df['Amplitude']==df.groupby('binned_freq')['Amplitude'].transform('max')]
    df = df[['Amplitude', 'binned_freq']].reset_index(drop=True)
    return df, df['Amplitude'].tolist()

#Idea --> Append df['Amplitude'] to another dataframe

cols = []
string = 'string'
for i in range(1000):
    if i % 20 == 0:
        string ='(' + str(i) + ',' + str(i+20) + ']'
        #Append in a list
        cols.append(string)

#Create Dataframe for appending df['Amplitude']
final_data = pd.DataFrame(columns=cols)
final_data_length = len(final_data)
final_data["gender"] = ""

#CLearly define what our data is --> final_data

def final_function(file, data):
    freq, mag = fft_new(file)
    sample, amp = freq_bin_vs_amp(freq, mag)
    #data = append_row(amp, data)
    to_append = amp
    gender = file[-13:-7]
    to_append.append(gender)
    df_length = len(data)
    data.loc[df_length] = to_append
    #data.iloc[-1:,-1:] = 
    return data

# This data would return the bucket wise max amplitude along with the gender 

