import streamlit as st
import numpy as np
import pandas as pd


st.set_page_config(
    page_title="Home page",
)

st.write("# NLP reviews analysis dashboard")


st.markdown(
    """
    The Data Science process was conducted following those steps:
    - **Extraction** of reviews both on **Google Play Store** and **Apple Store**
    - Translation of reviews 
    - Splitting of each sentences in reviews
    - **Sentiment Analysis** on each sentences to categorize between **POSITIVE**/**NEUTRAL**/**NEGATIVE**
    - Splitting of the whole dataset based on **categories**
    - **Clusterization** of **each** sentences
    - **Topic modelling** for **each** clusters
    - **Visualization** of **topics** 
    - **Creation** of this dashboard
    
    ### Pages:
    - Home page describing the **Data Science process**
    - High level **Results**
    - **In depth Visualization of Results**
"""
)