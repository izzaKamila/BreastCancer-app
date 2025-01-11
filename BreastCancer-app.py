#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  13 18:09:06 2024

@author: mac
"""

import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import numpy as np



# loading the saved models

nb_model = joblib.load('NB_joblib.sav')
dt_model = joblib.load('DT_joblib.sav')
rf_model = joblib.load('RF_joblib.sav')


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Breast Cancer Prediction System',
                          
                          ['Naive Bayes',
                           'Decision Tree',
                           'Random Forest'],
                          default_index=0)
    
    


#Breast Cancer with NB
if (selected == 'Naive Bayes'):
    
    #page title
    st.title('Breast Cancer Prediction')
    st.header('The Accuracy using Naive Bayes is 96.7%')
    st.write("")

    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Clump_thickness = st.number_input('Number of Clump Thickness')
        
    with col2:
        Uniformity_of_cell_size = st.number_input('Number of Size Cell')
    
    with col3:
        Uniformity_of_cell_shape = st.number_input('Number of Cell Shape')
    
    with col1:
        Marginal_adhesion = st.number_input('Marginal Adhesion value')
    
    with col2:
        Single_epithelial_cell_size = st.number_input('Epithelial value')
    
    with col3:
        Bare_nuclei = st.number_input('Bare Nuclei value')
    
    with col1:
        Bland_chromatin = st.number_input('Bland Chromatin value')
    
    with col2:
        Normal_nucleoli = st.number_input('Normal Nucleoli value')
    
    # code for Prediction
    cancer_type = ''
    
    # creating a button for Prediction
    if st.button('Breast Cancer Test Result'):
        # Mengubah input menjadi array 2D
        input_features = np.array([Clump_thickness, Uniformity_of_cell_size, Uniformity_of_cell_shape, Marginal_adhesion, Single_epithelial_cell_size,
                                   Bare_nuclei, Bland_chromatin, Normal_nucleoli]).reshape(1, -1)
        
        cancer_pred = nb_model.predict(input_features)
        
        if (cancer_pred[0] == 2):
            cancer_type = 'Benign'
        else:
            cancer_type = 'Malignant'
        
        st.success(cancer_type)




#Breast Cancer with DT
if (selected == 'Decision Tree'):
    
    #page title
    st.title('Breast Cancer Prediction')
    st.header('The Accuracy using Decision Tree is 89.1%')
    st.write("")
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Clump_thickness = st.number_input('Number of Clump Thickness')
        
    with col2:
        Uniformity_of_cell_size = st.number_input('Number of Size Cell')
    
    with col3:
        Uniformity_of_cell_shape = st.number_input('Number of Cell Shape')
    
    with col1:
        Marginal_adhesion = st.number_input('Marginal Adhesion value')
    
    with col2:
        Single_epithelial_cell_size = st.number_input('Epithelial value')
    
    with col3:
        Bare_nuclei = st.number_input('Bare Nuclei value')
    
    with col1:
        Bland_chromatin = st.number_input('Bland Chromatin value')
    
    with col2:
        Normal_nucleoli = st.number_input('Normal Nucleoli value')
    
    # code for Prediction
    cancer_type = ''
    
    # creating a button for Prediction
    if st.button('Breast Cancer Test Result'):
        # Mengubah input menjadi array 2D
        input_features = np.array([Clump_thickness, Uniformity_of_cell_size, Uniformity_of_cell_shape, Marginal_adhesion, Single_epithelial_cell_size,
                                   Bare_nuclei, Bland_chromatin, Normal_nucleoli]).reshape(1, -1)
        
        cancer_pred = dt_model.predict(input_features)
        
        if (cancer_pred[0] == 2):
            cancer_type = 'Benign'
        else:
            cancer_type = 'Malignant'
        
        st.success(cancer_type)


   
#Breast Cancer with RF
if (selected == 'Random Forest'):
    
    #page title
    st.title('Breast Cancer Prediction')
    st.header('The Accuracy using Random Forest is 94.9%')
    st.write("")
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Clump_thickness = st.number_input('Number of Clump Thickness')
        
    with col2:
        Uniformity_of_cell_size = st.number_input('Number of Size Cell')
    
    with col3:
        Uniformity_of_cell_shape = st.number_input('Number of Cell Shape')
    
    with col1:
        Marginal_adhesion = st.number_input('Marginal Adhesion value')
    
    with col2:
        Single_epithelial_cell_size = st.number_input('Epithelial value')
    
    with col3:
        Bare_nuclei = st.number_input('Bare Nuclei value')
    
    with col1:
        Bland_chromatin = st.number_input('Bland Chromatin value')
    
    with col2:
        Normal_nucleoli = st.number_input('Normal Nucleoli value')
    
        
    # code for Prediction
    cancer_type = ''
    
    # creating a button for Prediction
    if st.button('Breast Cancer Test Result'):
        # Mengubah input menjadi array 2D
        input_features = np.array([Clump_thickness, Uniformity_of_cell_size, Uniformity_of_cell_shape, Marginal_adhesion, Single_epithelial_cell_size,
                                   Bare_nuclei, Bland_chromatin, Normal_nucleoli]).reshape(1, -1)
        
        cancer_pred = rf_model.predict(input_features)
        
        if (cancer_pred[0] == 2):
            cancer_type = 'Benign'
        else:
            cancer_type = 'Malignant'
        
        st.success(cancer_type)
