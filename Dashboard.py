# app.py
import streamlit as st

st.set_page_config(
    page_title="Aplikasi Prediksi Harga Rumah Jabodetabek",
    layout="wide",
    initial_sidebar_state="expanded" # Sidebar akan terbuka secara default
)

st.markdown("<h1 style='color:#2E86C1;'>Selamat Datang di Aplikasi Prediksi Harga Rumah Jabodetabek</h1>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("## Anggota Kelompok")
st.write("""
-   **Muhammad Ardiansyah Firdaus** - 220102057
-   **Rifqi Aini Wahdaniyatillahi** - 220102073
-   **Akbar Pradhika Ashari** -  220102010
-   **Nizar Ahmad Baihaqi** - 220102066
""")

st.markdown("## Penjelasan Analisis Kami")
st.write("""
Aplikasi ini merupakan hasil proyek Data Mining yang bertujuan untuk memprediksi harga rumah
di wilayah Jabodetabek dan melakukan analisis klastering pada data properti.
Kami menggunakan berbagai teknik Machine Learning, mulai dari pemahaman data, pra-pemrosesan,
hingga pembangunan dan evaluasi model.
""")

st.markdown("---")
st.markdown("### Navigasi Aplikasi")
st.info("Silakan gunakan menu di sidebar sebelah kiri untuk menjelajahi berbagai tahapan proyek kami:")
st.write("""
-   **Business Understanding**: Penjelasan tentang tujuan bisnis dan pemahaman masalah.
-   **Data Understanding**: Eksplorasi awal dataset, struktur, dan kualitas data.
-   **Exploratory Data Analysis (EDA)**: Visualisasi dan analisis statistik untuk memahami pola data.
-   **Data Preparation**: Langkah-langkah pra-pemrosesan data seperti penanganan *missing values* dan *feature engineering*.
-   **Modeling**: Pembangunan dan evaluasi model regresi dan klastering.
-   **Analisis Klastering**: Halaman untuk menampilkan hasil klastering data properti.
-   **House Price Prediction**: Halaman untuk melakukan prediksi harga rumah berdasarkan input Anda.
""")

# Anda bisa menambahkan gambar, video, atau elemen lain di halaman dashboard ini.