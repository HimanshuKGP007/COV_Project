import time, os
import logging
import streamlit as st
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from settings import IMAGE_DIR, DURATION, WAVE_OUTPUT_FILE
DURATION = 3   # 3 seconds
from src.sound import sound
from src.model import CNN
from setup_logging import setup_logging
import csv
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

setup_logging()
logger = logging.getLogger('app')

df = pd.read_csv(r"C:\Users\DELL\COV_Project\Files\smote_no_encode.csv")
scaler = MinMaxScaler()
X = scaler.fit_transform(np.array(df.iloc[:, :-1], dtype = float))
features = ['chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth',
       'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4',
       'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11',
       'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18',
       'mfcc19', 'mfcc20']

X_s = pd.DataFrame(X, columns = features)
X_s['label'] = df['label']

# New Features
CLASSES = ['COVID Negative', 'COVID Positive']
CLASSES_MAP = {'COVID Negative':0, 'COVID Positive':1}

def predict(self, filepath, loadmodel=True):
        logger.info('Prediction')
        if loadmodel:
            # self.load_model()
            pass
        else:
            # try:
                y, sr = librosa.load(filepath, duration=3)
                ps = librosa.feature.melspectrogram(y=y, sr=sr)
                
                header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
                for i in range(1, 21):
                        header += f' mfcc{i}'
                header = header.split()

                file = open('test.csv', 'w')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(header)
                for i in range(1):
                    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                    rmse = librosa.feature.rms(y=y)
                    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                    zcr = librosa.feature.zero_crossing_rate(y)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr)
                    to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
                    for e in mfcc:
                        to_append += f' {np.mean(e)}'
                    file = open('test.csv', 'a')
                    with file:
                        writer = csv.writer(file)
                        writer.writerow(to_append.split())
                file.close()
                
                test = pd.read_csv('test.csv')
                test_normalised = scaler.transform(test)
                print(test_normalised) 
                logger.info("The recorded chord is " + test_normalised)
                #test_normalised=test_normalised[:,~np.all(np.isnan(d), axis=0)]
                loaded_model = pickle.load(open(r'C:\Users\DELL\COV_Project\models\finalized_model.sav', 'rb'))
                prediction = loaded_model.predict(test_normalised)
                class_id = prediction[0]
                chord = str(CLASSES[class_id])
                logger.info("The recorded chord is " + chord)

def init_model():
    cnn = pickle.load(open(r'C:\Users\DELL\COV_Project\models\finalized_model.sav', 'rb'))
    #cnn.load_model()
    return cnn

def main():
    title = "COVID-19 Detection"
    st.title(title)
    #image = Image.open(os.path.join(IMAGE_DIR, 'app_guitar.jpg'))
    #st.image(image, use_column_width=True)

    if st.button('Record'):
        with st.spinner(f'Recording for {DURATION} seconds ....'):
            sound.record()
        st.success("Recording completed")

    if st.button('Play'):
        # sound.play()
        try:
            audio_file = open(WAVE_OUTPUT_FILE, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
        except:
            st.write("Please record sound first")

    if st.button('Classify'):
        cnn = init_model()
        with st.spinner("Classifying the chord"):
            chord = cnn.predict(WAVE_OUTPUT_FILE, False)
        st.success("Classification completed")
        st.write("### The recorded chord is **", chord + "**")
        if chord == 'N/A':
            st.write("Please record sound first")
        st.write("\n")


if __name__ == '__main__':
    main()
    # for i in range(100):
    #   # Update the progress bar with each iteration.
    #   latest_iteration.text(f'Iteration {i+1}')
    #   bar.progress(i + 1)
    #   time.sleep(0.1)