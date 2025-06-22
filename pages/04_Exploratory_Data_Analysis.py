# pages/04_Exploratory_Data_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Exploratory Data Analysis",
    layout="wide"
)

st.markdown("<h1 style='color:#2E86C1;'>Exploratory Data Analysis (EDA)</h1>", unsafe_allow_html=True)
st.markdown("---")

st.write("Melakukan analisis statistik dan visualisasi untuk memahami karakteristik dan pola dalam data harga rumah.")

# Memuat dan pra-proses data seperti di notebook agar EDA konsisten
@st.cache_data # Cache data agar tidak diulang setiap kali halaman dimuat
def load_and_preprocess_data(file_path):
    df_eda = pd.read_csv(file_path)
    
    # Penanganan Missing Values (dari notebook cell 10)
    num_cols = df_eda.select_dtypes(include=[np.number]).columns
    df_eda[num_cols] = df_eda[num_cols].apply(lambda x: x.fillna(x.median()))
    cat_cols = df_eda.select_dtypes(include=['object']).columns
    df_eda[cat_cols] = df_eda[cat_cols].apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))
    
    # Feature Engineering (dari notebook cell 11)
    df_eda['price_per_m2'] = df_eda['price_in_rp'] / df_eda['land_size_m2']
    df_eda['total_rooms'] = df_eda['bedrooms'] + df_eda['bathrooms'] + df_eda['maid_bedrooms'] + df_eda['maid_bathrooms']
    
    def age_category(age):
        if pd.isnull(age):
            return 'unknown'
        elif age <= 2:
            return 'baru'
        elif age <= 10:
            return 'sedang'
        else:
            return 'lama'
    df_eda['house_age_category'] = df_eda['building_age'].apply(age_category)
    return df_eda

df_eda = load_and_preprocess_data('jabodetabek_house_price.csv')

if df_eda is not None:
    st.subheader("1. Statistik Deskriptif")
    st.write("Statistik deskriptif untuk kolom numerik:")
    st.dataframe(df_eda.describe())
    st.write("Statistik deskriptif untuk kolom kategorikal:")
    st.dataframe(df_eda.describe(include='object'))

    st.subheader("2. Distribusi Harga Rumah")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.histplot(df_eda['price_in_rp'], bins=50, kde=True, ax=ax1)
    ax1.set_title('Distribusi Harga Rumah')
    ax1.set_xlabel('Harga (Rp)')
    ax1.set_ylabel('Jumlah')
    st.pyplot(fig1)
    st.write("Visualisasi ini menunjukkan distribusi harga rumah. Tampak ada *outlier* dengan harga sangat tinggi.")

    st.subheader("3. Matriks Korelasi Fitur Numerik")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_eda.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', ax=ax2)
    ax2.set_title('Heatmap Korelasi Fitur Numerik')
    st.pyplot(fig2)
    st.write("Heatmap ini menggambarkan korelasi antara fitur-fitur numerik. Fitur dengan korelasi tinggi terhadap harga (`price_in_rp`) adalah `building_size_m2`, `land_size_m2`, dan `price_per_m2`.")

    st.subheader("4. Distribusi Harga Berdasarkan Kota")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='city', y='price_in_rp', data=df_eda, ax=ax3)
    ax3.set_title('Distribusi Harga Rumah per Kota')
    ax3.set_xlabel('Kota')
    ax3.set_ylabel('Harga (Rp)')
    st.pyplot(fig3)
    st.write("Grafik ini menunjukkan variasi harga rumah antar kota. Beberapa kota mungkin memiliki median harga yang lebih tinggi atau sebaran harga yang lebih luas.")

    # Tambahkan visualisasi lain yang relevan dari notebook Anda
    # Contoh: Distribusi Usia Bangunan per Kategori
    st.subheader("5. Distribusi Usia Bangunan")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.countplot(x='house_age_category', data=df_eda, ax=ax4, palette='viridis')
    ax4.set_title('Distribusi Kategori Usia Rumah')
    ax4.set_xlabel('Kategori Usia Rumah')
    ax4.set_ylabel('Jumlah Properti')
    st.pyplot(fig4)
    st.write("Mayoritas properti berada dalam kategori 'baru' dan 'sedang'.")