import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np 
from settings import IMAGE_DIR, DURATION, WAVE_OUTPUT_FILE

st.title("COVID-19 Detector")

nav = st.sidebar.radio("Navigation",["Testing","Contribute"])
# Trying to add Audio Record feature
# if nav == "Contribute":
#     st.header("Contribute to our dataset")
#     ex = st.number_input("Enter your Name",0.0,20.0) # Name_imput
#     sal = st.number_input("Enter your Salary",0.00,1000000.00,step = 1000.0)# Covid Positive or healthy -- Dropdown box
#     if st.button("submit"):
#         to_add = {"Name":[ex],"Salary":[sal]}
#         to_add = pd.DataFrame(to_add)
#         to_add.to_csv("data//Salary_Data.csv",mode='a',header = False,index= False)
#         st.success("Submitted")
def main():

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

    # Add a placeholder
    # if st.button('Display Spectrogram'):
    #     # type = st.radio("Scale of spectrogram:",
    #     #                 ('mel', 'DB'))
    #     if os.path.exists(WAVE_OUTPUT_FILE):
    #         spectrogram, format = get_spectrogram(type='mel')
    #         display(spectrogram, format)
    #     else:
    #         st.write("Please record sound first")