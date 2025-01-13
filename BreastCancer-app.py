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
import sklearn
print(sklearn.__version__)

required_files = ['NB2_model.sav', 'DT2_model.sav', 'RF2_model.sav']
missing_files = [file for file in required_files if not os.path.exists(file)]

if missing_files:
    st.error(f"Missing files: {', '.join(missing_files)}. Please upload these files to the project directory.")
    st.stop()

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
try:
    # Memuat semua model pada awal
    nb_model = pickle.load(open('NB2_model.sav', 'rb'))
    dt_model = pickle.load(open('DT2_model.sav', 'rb'))
    rf_model = pickle.load(open('RF2_model.sav', 'rb'))

    # Debugging tipe model (log ke terminal/file)
    logging.info("Type of nb_model: %s", type(nb_model))
    logging.info("Type of dt_model: %s", type(dt_model))
    logging.info("Type of rf_model: %s", type(rf_model))
except FileNotFoundError:
    st.error("""
    Model files not found. Please ensure these files exist in the root directory:
    - NB2_model.sav
    - DT2_model.sav
    - RF2_model.sav
    """)
    st.stop()

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Breast Cancer Prediction System',          
                          ['Naive Bayes', 'Decision Tree', 'Random Forest'],
                          default_index=0)

# Menampilkan halaman berdasarkan pilihan model
if selected == 'Naive Bayes':
    st.title('Breast Cancer Prediction')
    st.header('The Accuracy using Naive Bayes is 96.7%')
    st.write("")

    # Menampilkan tipe model untuk verifikasi
    st.write("Model Type: ", type(nb_model))

elif selected == 'Decision Tree':
    st.title('Breast Cancer Prediction')
    st.header('The Accuracy using Decision Tree is 89.1%')
    st.write("")

    # Menampilkan tipe model untuk verifikasi
    st.write("Model Type: ", type(dt_model))

elif selected == 'Random Forest':
    st.title('Breast Cancer Prediction')
    st.header('The Accuracy using Random Forest is 94.9%')
    st.write("")

    # Menampilkan tipe model untuk verifikasi
    st.write("Model Type: ", type(rf_model))

# Form Input untuk semua model
col1, col2, col3 = st.columns(3)

with col1:
    Clump_thickness = st.number_input('Clump Thickness', min_value=0.0, max_value=10.0, step=0.1)
with col2:
    Uniformity_of_cell_size = st.number_input('Uniformity of Cell Size', min_value=0.0, max_value=10.0, step=0.1)
with col3:
    Uniformity_of_cell_shape = st.number_input('Uniformity of Cell Shape', min_value=0.0, max_value=10.0, step=0.1)
with col1:
    Marginal_adhesion = st.number_input('Marginal Adhesion', min_value=0.0, max_value=10.0, step=0.1)
with col2:
    Single_epithelial_cell_size = st.number_input('Single Epithelial Cell Size', min_value=0.0, max_value=10.0, step=0.1)
with col3:
    Bare_nuclei = st.number_input('Bare Nuclei', min_value=0.0, max_value=10.0, step=0.1)
with col1:
    Bland_chromatin = st.number_input('Bland Chromatin', min_value=0.0, max_value=10.0, step=0.1)
with col2:
    Normal_nucleoli = st.number_input('Normal Nucleoli', min_value=0.0, max_value=10.0, step=0.1)

# Code for Prediction
cancer_type = ''

# Prediction Button
if st.button('Breast Cancer Test Result'):
    # Mengubah input menjadi array 2D (pastikan input adalah array 2D)
    input_features = np.array([[Clump_thickness, Uniformity_of_cell_size, Uniformity_of_cell_shape,
                                Marginal_adhesion, Single_epithelial_cell_size, Bare_nuclei,
                                Bland_chromatin, Normal_nucleoli]])

    # Periksa ukuran input untuk memastikan input adalah array 2D
    st.write("Input Features Shape: ", input_features.shape)

    # Melakukan prediksi berdasarkan model yang dipilih
    if selected == 'Naive Bayes':
        try:
            cancer_pred = nb_model.predict(input_features)
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            st.stop()
    elif selected == 'Decision Tree':
        try:
            cancer_pred = dt_model.predict(input_features)
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            st.stop()
    elif selected == 'Random Forest':
        try:
            cancer_pred = rf_model.predict(input_features)
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            st.stop()

    # Menentukan jenis kanker
    if cancer_pred[0] == 2:
        cancer_type = 'Benign'
    else:
        cancer_type = 'Malignant'

    st.success(f"Prediksi: {cancer_type}")
