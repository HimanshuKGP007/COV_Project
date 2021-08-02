# Improt dependencies
import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np


# Page set-up
st.set_page_config( page_title="Virufy Dashboard", 
                    layout='centered',
                    page_icon = 'Virufy_Icon.png'
                    )

st.image('Virufy_Icon.png')

                
st.title("How well do our models predict Covid?") # Mian title

# Data prep
results_df = pd.read_csv("results.csv") # load data
country_list = list(results_df.country.unique()) # get the list name of country
country_list.append('All Countries') # Add 'All' option
# model_list = ["Ahmed_model", "model_2", "model_3", "model_4", "model_5"] # get the list name of model
model_list  = list(results_df.columns[5:]) 


country = st.sidebar.selectbox("Country",options=country_list)


## By country

if country == 'All Countries':
    df_country = results_df # All result
else:
    df_country = results_df[results_df['country']==country] # By country

Source_list = df_country['source'].unique() # Incase certain source not include all countries
Source = st.sidebar.selectbox("Source",options=Source_list)

## Then by Source

df_source = df_country[df_country['source']==Source]




## If displaying all countires result
if country == 'All Countries' :
    ### Create dataframes for each country
    country_list_select = list(df_source.country.unique()) 
    dict_countries={i:pd.DataFrame() for i in country_list_select} 

    ### for each country, cache the pcr_test_result 
    for country_name in dict_countries: 
        df_all = df_source[df_source['country']==country_name]
        result = df_all.pcr_test_result
        #### for each model within each country: compute the confusion matrix and indicators  
        for model_name in model_list:
           
            pred = df_all[model_name]

            T_N, F_P, F_N, T_P = confusion_matrix(result,pred).ravel()

            
            Sens = (T_P/(T_P+F_N))
            Spec = (T_N/(T_N+F_P))
            Ppv = (T_P/(T_P+F_P))
            Npv = (T_N/(F_N+T_N))
            F1_Score = f1_score(result, pred)
            dict_countries[country_name][model_name]= pd.Series([Sens,Spec,Ppv,Npv,F1_Score])
            
            
        ### display the dataframe for each country
        df_index = ["Sensitivity", "Specificity", "PPV", "NPV","F1_Score"]
        dict_countries[country_name]['Indicator'] = df_index
        dict_countries[country_name] = dict_countries[country_name].set_index('Indicator') # Set index
        st.subheader(f"Model robustness: {country_name}")
        st.dataframe(dict_countries[country_name].style.highlight_max(axis=1))

else:


    ## Then select by model
    
    model = st.sidebar.selectbox("Model",options=model_list)

    pcr_test_result_pred = df_source[model]

    df_model = df_source[df_source.index == df_source[model].index]  # Incase some country or sourcse do not have model predicted




    # Then by age

    age_min = int(df_model.age.min())
    age_max = int(df_model.age.max())

    
    if age_min==age_max:

        df_age = df_model # If the age is only one , then hide the age selection bar
    else:
        
        age = st.sidebar.slider("Age Range", age_min,age_max,value =(age_min,age_max),step=5) # return as int/float/date/time/datetime or tuple of int/float/date/time/datetime
        df_age = df_model[(df_model['age']>age[0]) & (df_model['age']<age[1])] # else slicing the df by range of the age 
        # Else display country by country

    # DataFrames
    st.subheader(f"Results for: {country}")
    df = pd.DataFrame()

    pcr_test_result = df_age['pcr_test_result']



    for i in model_list:
        # prediction
        pred = df_age[i]
        # confusion matrix
        T_N, F_P, F_N, T_P = confusion_matrix(pcr_test_result,pred).ravel()

            

        #Sensitivity, Specificity, PPV, NPV
        Sens = (T_P/(T_P+F_N))
        Spec = (T_N/(T_N+F_P))
        Ppv = (T_P/(T_P+F_P))
        Npv = (T_N/(F_N+T_N))
        F1_Score = f1_score(pcr_test_result, pred)
        

        # Each model assign the Sensitivity, Specificity, PPV, NPV, F1_Score
        df[i]= pd.Series([Sens,Spec,Ppv,Npv,F1_Score])

    df_index = ["Sensitivity", "Specificity", "PPV", "NPV", "F1_Score"]
    df['indicator'] = df_index
    df = df.set_index('indicator') # Set index

    st.dataframe(df.style.highlight_max(axis=1)) # Display the dataframe


# Pie Chart & Bar Graph
    ## Confusion Matrix
    pcr_test_result_pred = df_age[model]

    TN,FP,FN,TP = confusion_matrix(pcr_test_result, pcr_test_result_pred).ravel()

    # Title
    st.subheader(f"Distribution for: {country}")


    
    # Pie Chart
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].pie([FN,TN,TP,FP], labels= ['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'], autopct='%1.1f%%')
    



    # Bar Chart
    ax[1].bar(['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'],[FN,TN,TP,FP])
    ax[1].tick_params(labelsize=8)
    st.pyplot(fig)

    