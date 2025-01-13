#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  13 18:09:06 2024

@author: mac
"""

import os
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pickle
import logging

required_files = ['NB_model.sav', 'DT_model.sav', 'RF_model.sav']
missing_files = [file for file in required_files if not os.path.exists(file)]

if missing_files:
    st.error(f"Missing files: {', '.join(missing_files)}. Please upload these files to the project directory.")
    st.stop()


# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
try:
    nb_model = pickle.load(open('NB_model.sav', 'rb'))
    dt_model = pickle.load(open('DT_model.sav', 'rb'))
    rf_model = pickle.load(open('RF_model.sav', 'rb'))

    # Debugging tipe model (log ke terminal/file)
    logging.info("Type of nb_model: %s", type(nb_model))
    logging.info("Type of dt_model: %s", type(dt_model))
    logging.info("Type of rf_model: %s", type(rf_model))
except FileNotFoundError:
    st.error("""
    Model files not found. Please ensure these files exist in the root directory:
    - NB_model.sav
    - DT_model.sav
    - RF_model.sav
    """)
    st.stop()

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
        Clump_thickness = st.number_input('Number of Clump Thickness', min_value=0.0, max_value=10.0, step=0.1)
        
    with col2:
        Uniformity_of_cell_size = st.number_input('Number of Size Cell', min_value=0.0, max_value=10.0, step=0.1)
    
    with col3:
        Uniformity_of_cell_shape = st.number_input('Number of Cell Shape', min_value=0.0, max_value=10.0, step=0.1)
    
    with col1:
        Marginal_adhesion = st.number_input('Marginal Adhesion value', min_value=0.0, max_value=10.0, step=0.1)
    
    with col2:
        Single_epithelial_cell_size = st.number_input('Epithelial value', min_value=0.0, max_value=10.0, step=0.1)
    
    with col3:
        Bare_nuclei = st.number_input('Bare Nuclei value', min_value=0.0, max_value=10.0, step=0.1)
    
    with col1:
        Bland_chromatin = st.number_input('Bland Chromatin value', min_value=0.0, max_value=10.0, step=0.1)
    
    with col2:
        Normal_nucleoli = st.number_input('Normal Nucleoli value', min_value=0.0, max_value=10.0, step=0.1)
    
    # code for Prediction
    cancer_type = ''
    
    # creating a button for Prediction
    if st.button('Breast Cancer Test Result'):
        # Mengubah input menjadi array 2D
        input_features = np.array([Clump_thickness, Uniformity_of_cell_size, Uniformity_of_cell_shape, Marginal_adhesion, Single_epithelial_cell_size,
                                   Bare_nuclei, Bland_chromatin, Normal_nucleoli]).reshape(1, -1)

        with open('NB_model.sav', 'rb') as file: 
            nb_model = pickle.load(file)
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
        Clump_thickness = st.number_input('Number of Clump Thickness', min_value=0.0, max_value=10.0, step=0.1)
        
    with col2:
        Uniformity_of_cell_size = st.number_input('Number of Size Cell', min_value=0.0, max_value=10.0, step=0.1)
    
    with col3:
        Uniformity_of_cell_shape = st.number_input('Number of Cell Shape', min_value=0.0, max_value=10.0, step=0.1)
    
    with col1:
        Marginal_adhesion = st.number_input('Marginal Adhesion value', min_value=0.0, max_value=10.0, step=0.1)
    
    with col2:
        Single_epithelial_cell_size = st.number_input('Epithelial value', min_value=0.0, max_value=10.0, step=0.1)
    
    with col3:
        Bare_nuclei = st.number_input('Bare Nuclei value', min_value=0.0, max_value=10.0, step=0.1)
    
    with col1:
        Bland_chromatin = st.number_input('Bland Chromatin value', min_value=0.0, max_value=10.0, step=0.1)
    
    with col2:
        Normal_nucleoli = st.number_input('Normal Nucleoli value', min_value=0.0, max_value=10.0, step=0.1)
    
    # code for Prediction
    cancer_type = ''
    
    # creating a button for Prediction
    if st.button('Breast Cancer Test Result'):
        # Mengubah input menjadi array 2D
        input_features = np.array([Clump_thickness, Uniformity_of_cell_size, Uniformity_of_cell_shape, Marginal_adhesion, Single_epithelial_cell_size,
                                   Bare_nuclei, Bland_chromatin, Normal_nucleoli]).reshape(1, -1)
       
        with open('DT_model.sav', 'rb') as file: 
            dt_model = pickle.load(file)        
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
        Clump_thickness = st.number_input('Number of Clump Thickness', min_value=0.0, max_value=10.0, step=0.1)
        
    with col2:
        Uniformity_of_cell_size = st.number_input('Number of Size Cell', min_value=0.0, max_value=10.0, step=0.1)
    
    with col3:
        Uniformity_of_cell_shape = st.number_input('Number of Cell Shape', min_value=0.0, max_value=10.0, step=0.1)
    
    with col1:
        Marginal_adhesion = st.number_input('Marginal Adhesion value', min_value=0.0, max_value=10.0, step=0.1)
    
    with col2:
        Single_epithelial_cell_size = st.number_input('Epithelial value',  min_value=0.0, max_value=10.0, step=0.1)
    
    with col3:
        Bare_nuclei = st.number_input('Bare Nuclei value',  min_value=0.0, max_value=10.0, step=0.1)
    
    with col1:
        Bland_chromatin = st.number_input('Bland Chromatin value',  min_value=0.0, max_value=10.0, step=0.1)
    
    with col2:
        Normal_nucleoli = st.number_input('Normal Nucleoli value',  min_value=0.0, max_value=10.0, step=0.1)
    
        
    # code for Prediction
    cancer_type = ''
    
    # creating a button for Prediction
    if st.button('Breast Cancer Test Result'):
        # Mengubah input menjadi array 2D
        input_features = np.array([Clump_thickness, Uniformity_of_cell_size, Uniformity_of_cell_shape, Marginal_adhesion, Single_epithelial_cell_size,
                                   Bare_nuclei, Bland_chromatin, Normal_nucleoli]).reshape(1, -1)

        with open('RF_model.sav', 'rb') as file: 
            rf_model = pickle.load(file)    
        cancer_pred = rf_model.predict(input_features)
        
        if (cancer_pred[0] == 2):
            cancer_type = 'Benign'
        else:
            cancer_type = 'Malignant'
        
        st.success(cancer_type)


