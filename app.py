import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle 
import plotly.graph_objects as go 
# Page configuration st.set_page_config( 
page_title='Customer Churn Predictor', page_icon=' ', layout='wide' 
) st.title(' Customer Churn Prediction System')
@st.cache_resource def load_model(): with open('best_churn_model.pkl', 
'rb') as file: 
  model = pickle.load(file) return model model = 
load_model() st.success(' Model loaded successfully!')
