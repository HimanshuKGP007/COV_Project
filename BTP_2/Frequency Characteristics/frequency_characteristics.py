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

def custom_fft(y, fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    vals = 2.0/N * np.abs(yf[0:N//2])  
    return xf, vals

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

file = r'C:\Users\DELL\COV_Project\Files\cough_wavs_sample\cleaned_data\Positive\1192_Positive_male_28.wav'
freq, mag = fft_new(file)

def createSeries (series_list):
    series_list = pd.Series(series_list)  
    return series_list

def freq_amp(freq, mag):
    frequencies = createSeries(freq) 
    Amplitude = createSeries(mag)
    data = {"frequencies": frequencies,
            "Amplitude": Amplitude}
    df1 = pd.concat(data, axis = 1)
    df2 = df1[df1['frequencies']<1000]
    return df2

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def raw_to_agg(df2):
    squares1 = []
    for i in range(51):
        squares1.append(i * 20)

    cols = []
    string = 'string'
    for i in range(1000):
        if i % 20 == 0:
           string ='(' + str(i) + ',' + str(i+20) + ']'
        #Append in a list
        cols.append(string)

    a = pd.cut(df2.frequencies,bins = squares1, labels = cols)
    # a contains the respective bins for all frequency values in data
    return a

def a_to_amp(a, df2):
    freq = a
    amp = df2.Amplitude
    data = {"freq": freq, "amp": amp}
    df3 = pd.concat(data, axis = 1)
    final_data_single_rec = df3.groupby(['freq']).agg({'amp': [percentile(80)]})
    final_data_single_rec.columns = ['_'.join(col) for col in final_data_single_rec.columns.values]
    mn = final_data_single_rec.to_string(index=False, header= False)
    my_list = mn.split("\n")
    a1 = my_list
    b =  [float(float(s)) for s in a1]
    amp = b
    return amp

def gen_add(amp, file):
    gender = file[-13:-7]
    gen = 0
    if gender == 'e_male':
        gen = 1
    gen = float(gen)
    amp.append(gen)
    return amp

def final_function(file, data):
    freq, mag = fft_new(file)
    df2 = freq_amp(freq, mag)
    amp = raw_to_agg(df2)
    amp = a_to_amp(amp, df2)
    #print(amp)
    amp = gen_add(amp, file)
    df_length = len(data)
    data.loc[df_length]=pd.DataFrame(amp).iloc[df_length]
    #print(df_length)
    data.loc[df_length, 'gender'] = file[-13:-7]
    data.loc[df_length] = amp
    return data

#Test
# file =r'C:\Users\DELL\COV_Project\Files\cough_wavs_sample\cleaned_dataPositive\1192_Positive_male_28.wav'
# final_data = final_function(file, final_data)