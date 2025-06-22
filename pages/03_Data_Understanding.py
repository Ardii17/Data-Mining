# pages/03_Data_Understanding.py
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Data Understanding",
    layout="wide"
)

st.markdown("<h1 style='color:#2E86C1;'>Data Understanding</h1>", unsafe_allow_html=True)
st.markdown("---")

st.subheader("1. Memuat Dataset")
st.write("Dataset harga rumah Jabodetabek dimuat untuk analisis.")
try:
    df_raw = pd.read_csv('jabodetabek_house_price.csv') # Muat ulang dataset
    st.write("Dataset berhasil dimuat!")
    st.dataframe(df_raw.head())
except FileNotFoundError:
    st.error("File 'jabodetabek_house_price.csv' tidak ditemukan. Pastikan ada di direktori yang sama.")
    df_raw = None # Set to None if file not found

if df_raw is not None:
    st.subheader("2. Informasi Dataset")
    st.write("Berikut adalah ringkasan informasi dataset:")
    st.text(df_raw.info(verbose=True, buf=None)) # Menggunakan st.text untuk menampilkan info()

    st.subheader("3. Ukuran Dataset")
    st.write(f"Dataset memiliki **{df_raw.shape[0]} baris** dan **{df_raw.shape[1]} kolom**.")

    st.subheader("4. Missing Values")
    st.write("Jumlah *missing values* (nilai hilang) per kolom:")
    missing_values = df_raw.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        st.dataframe(missing_values.rename('Jumlah Missing Values'))
    else:
        st.info("Tidak ada *missing values* di dataset pada tahap ini.")

    st.subheader("5. Duplikasi Data")
    st.write("Jumlah data duplikat (baris identik):")
    duplicated_count = df_raw.duplicated().sum()
    st.write(f"Jumlah data duplikat: **{duplicated_count}**")
    if duplicated_count > 0:
        st.write("Contoh baris duplikat:")
        st.dataframe(df_raw[df_raw.duplicated()].head())
    else:
        st.info("Tidak ada data duplikat di dataset.")