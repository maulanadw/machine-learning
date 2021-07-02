import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Prediksi Harga Rumah App

Tampilkan prediksi harga perumahan!

""")