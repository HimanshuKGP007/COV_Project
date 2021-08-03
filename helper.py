import matplotlib.pyplot as plt
import librosa
from pathlib import Path
#from encoder.inference import plot_embedding_as_heatmap
import sounddevice as sd
import wavio
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

features = ['chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth',
       'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4',
       'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11',
       'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18',
       'mfcc19', 'mfcc20']

def preprocess(fn_wav):
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

def scaler_transform(feature):
    df = pd.read_csv(r"C:\Users\DELL\COV_Project\Files\smote_no_encode.csv")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(np.array(df.iloc[:, :-1]))
    X_s = pd.DataFrame(X, columns = features)
    X_s['label'] = df['label']
    test_normalised = scaler.transform(feature)
    return test_normalised

def draw_embed(embed, name, which):
    """
    Draws an embedding.

    Parameters:
        embed (np.array): array of embedding

        name (str): title of plot


    Return:
        fig: matplotlib figure
    """
    fig, embed_ax = plt.subplots()
    plot_embedding_as_heatmap(embed)
    embed_ax.set_title(name)
    embed_ax.set_aspect("equal", "datalim")
    embed_ax.set_xticks([])
    embed_ax.set_yticks([])
    embed_ax.figure.canvas.draw()
    return fig

# def preprocess(fn_wav):
#     y, sr = librosa.load(fn_wav, mono=True, duration=5)
#     chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#     rmse = librosa.feature.rms(y=y)
#     spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
#     spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#     rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#     zcr = librosa.feature.zero_crossing_rate(y)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
#     feature_row = {        
#         'chroma_stft': np.mean(chroma_stft),
#         'rmse': np.mean(rmse),
#         'spectral_centroid': np.mean(spectral_centroid),
#         'spectral_bandwidth': np.mean(spectral_bandwidth),
#         'rolloff': np.mean(rolloff),
#         'zero_crossing_rate': np.mean(zcr),        
#     }
#     for i, c in enumerate(mfcc):
#         feature_row[f'mfcc{i+1}'] = np.mean(c)

#     return feature_row

def create_spectrogram(voice_sample):
    """
    Creates and saves a spectrogram plot for a sound sample.

    Parameters:
        voice_sample (str): path to sample of sound

    Return:
        fig
    """

    in_fpath = Path(voice_sample.replace('"', "").replace("'", ""))
    original_wav, sampling_rate = librosa.load(str(in_fpath))

    # Plot the signal read from wav file
    fig = plt.figure()
    #plt.subplot(111)
    plt.title(f"Spectrogram of file {voice_sample}")

    plt.plot(original_wav)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    # plt.subplot(212)
    # plt.specgram(original_wav, Fs=sampling_rate)
    # plt.xlabel("Time")
    # plt.ylabel("Frequency")
    # # plt.savefig(voice_sample.split(".")[0] + "_spectogram.png")
    return fig

def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes

def record(duration=5, fs=22050):
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    sd.wait(duration)
    return myrecording

def save_record(path_myrecording, myrecording, fs):
    wavio.write(path_myrecording, myrecording, fs, sampwidth=2)
    return None
