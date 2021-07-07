import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

st.write("""
# Prediksi Harga Rumah App

Tampilkan prediksi *harga perumahan*!

""")
st.write('---')

# Load dan tampilkan datasets
# Membaca dataset
dataset = load_boston()
df = pd.DataFrame(dataset.data)
# Tetapkan kolom ke df
df.columns = dataset.feature_names
# Tetapkan variabel harga rumah
df["HARGA"] = dataset.target

# Tampilkan data tabel
if st.checkbox('Tampilkan dataset sebagai tabel'):
	st.dataframe(df)