import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
menu = ['Home','Predict']
choice = st.sidebar.selectbox('Menu',menu)

if choice == 'Home':
    st.header('Lili\'s Final Project')
    st.image('Photos\dataset-cover.png')
    st.subheader('Customer Personality Analysis and Prediction')
    st.info('Customer personality analysis helps a business to modify its product based on its target customers from different types of customer segments.')
    st.info('For example, instead of spending money to market a new product to every customer in the company’s database, a company can analyze which customer segment is most likely to buy the product and then market the product only on that particular segment.')
    st.balloons()

elif choice == 'Predict':
    st.image('Photos\mkt cmp.png')
    st.subheader('Predict if Customer accepts at least 1 Marketing campaign')
    
    #get dataset
    data = pd.read_csv('Dataset_cleaned\data_CusPers_cleaned_WithID.csv')
    
    #load pre-trained model
    classifier = pickle.load(open('Model\ROCAUC_XGB_76.sav','rb'))

    #input User's ID
    inp = int(st.number_input('Enter user ID',min_value=0,max_value=11191))
    df_user = data[data.ID == inp]
    df_user_input = df_user.drop('ID',axis=1)

    #apply model to make predictions
    prediction = classifier.predict(df_user_input)
    prediction_proba = classifier.predict_proba(df_user_input)
    
    #Show model's prediction and true label       
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Prediction')
        acceptance = np.array(['Not Accepted','Accepted'])
        st.write(acceptance[prediction])
    with col2:
        st.subheader('Label')
        acceptance = np.array(['Not Accepted','Accepted'])
        st.write(acceptance[int(df_user['AtLeast1Cmp'].values)])
    
    st.subheader('Customer\'s Details')
    st.write(df_user)
    #Show prediction probability
    st.subheader('Prediction Probability')
    st.write('0 stands for Not accepted, 1 stands for Accepted')
    st.write(pd.DataFrame(prediction_proba).applymap(lambda prediction_proba: '{:.2%}'.format(prediction_proba)).values)
    st.balloons()