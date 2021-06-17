import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np 


st.title("COVID-19 Detector")

nav = st.sidebar.radio("Navigation",["Testing","Contribute"])
# Trying to add Audio Record feature
if nav == "Contribute":
    st.header("Contribute to our dataset")
    ex = st.number_input("Enter your Name",0.0,20.0) # Name_imput
    sal = st.number_input("Enter your Salary",0.00,1000000.00,step = 1000.0)# Covid Positive or healthy -- Dropdown box
    if st.button("submit"):
        to_add = {"Name":[ex],"Salary":[sal]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("data//Salary_Data.csv",mode='a',header = False,index= False)
        st.success("Submitted")