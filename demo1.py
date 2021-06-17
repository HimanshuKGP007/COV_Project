import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np 


st.title("COVID-19 Detector")

nav = st.sidebar.radio("Navigation",["Testing","Contribute"])