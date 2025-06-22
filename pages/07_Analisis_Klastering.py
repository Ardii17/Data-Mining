# pages/08_Analisis_Klastering.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans # Needed for KMeans

st.set_page_config(
    page_title="Analisis Klastering",
    layout="wide"
)

st.markdown("<h1 style='color:#2E86C1;'>Analisis Klastering Data Rumah</h1>", unsafe_allow_html=True)
st.markdown("---")
st.write("Halaman ini menampilkan hasil analisis klastering pada data rumah untuk mengidentifikasi segmen-segmen properti yang berbeda.")

# Load and preprocess data for clustering
@st.cache_data # Cache data to avoid re-running every time the page is loaded
def load_and_cluster_data(file_path, preprocessor_path, kmeans_model_path):
    df_cluster_analysis = pd.read_csv(file_path)
    
    # Replicate preprocessing and feature engineering from notebook
    num_cols_analysis = df_cluster_analysis.select_dtypes(include=[np.number]).columns
    df_cluster_analysis[num_cols_analysis] = df_cluster_analysis[num_cols_analysis].apply(lambda x: x.fillna(x.median()))
    cat_cols_analysis = df_cluster_analysis.select_dtypes(include=['object']).columns
    df_cluster_analysis[cat_cols_analysis] = df_cluster_analysis[cat_cols_analysis].apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))

    df_cluster_analysis['price_per_m2'] = df_cluster_analysis['price_in_rp'] / df_cluster_analysis['land_size_m2']
    df_cluster_analysis['total_rooms'] = df_cluster_analysis['bedrooms'] + df_cluster_analysis['bathrooms'] + df_cluster_analysis['maid_bedrooms'] + df_cluster_analysis['maid_bathrooms']
    
    def age_category(age):
        if pd.isnull(age):
            return 'unknown'
        elif age <= 2:
            return 'baru'
        elif age <= 10:
            return 'sedang'
        else:
            return 'lama'
    df_cluster_analysis['house_age_category'] = df_cluster_analysis['building_age'].apply(age_category)

    # Prepare X for clustering (dropping the target 'price_in_rp')
    X_clustering_features_raw = df_cluster_analysis.drop('price_in_rp', axis=1)

    # Load the preprocessor
    loaded_preprocessor = joblib.load(preprocessor_path)
    X_cluster_processed = loaded_preprocessor.transform(X_clustering_features_raw)

    # Load the KMeans model
    loaded_kmeans_model = joblib.load(kmeans_model_path)
    
    cluster_labels = loaded_kmeans_model.predict(X_cluster_processed)
    df_cluster_analysis['cluster'] = cluster_labels
    return df_cluster_analysis

try:
    df_clustered = load_and_cluster_data('jabodetabek_house_price.csv', 'preprocessor.pkl', 'kmeans_cluster_model.pkl')

    st.subheader("1. Distribusi Data per Klaster")
    st.write("Jumlah properti yang termasuk dalam setiap klaster:")
    st.dataframe(df_clustered['cluster'].value_counts().rename('Jumlah Data per Klaster'))
    
    fig_count_dist = plt.figure(figsize=(8, 5))
    sns.countplot(x='cluster', data=df_clustered, palette='viridis')
    plt.title('Jumlah Data per Klaster')
    plt.xlabel('Klaster')
    plt.ylabel('Jumlah')
    st.pyplot(fig_count_dist)

    st.subheader("2. Statistik Deskriptif Tiap Klaster")
    st.write("Rata-rata fitur-fitur penting untuk setiap klaster:")
    
    # Define numerical columns for statistical summary
    numerical_cols_for_summary = [
        'price_in_rp', 'land_size_m2', 'building_size_m2', 'bedrooms',
        'bathrooms', 'total_rooms', 'price_per_m2', 'floors', 'building_age', 'garages'
    ]
    
    cluster_summary_df = df_clustered.groupby('cluster')[numerical_cols_for_summary].agg(['count', 'mean', 'median', 'min', 'max'])
    st.dataframe(cluster_summary_df.style.format({
        ('price_in_rp', 'mean'): "Rp.{:,.0f}", ('price_in_rp', 'median'): "Rp.{:,.0f}",
        ('price_in_rp', 'min'): "Rp.{:,.0f}", ('price_in_rp', 'max'): "Rp.{:,.0f}",
        ('price_per_m2', 'mean'): "{:,.2f}", ('price_per_m2', 'median'): "{:,.2f}"
    }))

    st.subheader("3. Karakteristik Klaster")
    st.markdown("""
        Berdasarkan analisis statistik di atas:
        * **Klaster 0**: Cenderung berisi properti dengan harga lebih rendah, ukuran tanah dan bangunan yang lebih kecil, serta jumlah kamar yang standar. Ini mungkin mewakili segmen pasar **properti *entry-level*** atau **menengah bawah**.
        * **Klaster 1**: Merupakan klaster yang sangat kecil (hanya 4 data), namun dengan harga, luas tanah, dan luas bangunan **jauh di atas rata-rata klaster lain**. Klaster ini kemungkinan besar mewakili **properti mewah atau *outlier*** di pasar.
        * **Klaster 2**: Berisi properti dengan harga, luas tanah, dan luas bangunan yang lebih tinggi dibandingkan Klaster 0, namun lebih masuk akal dibandingkan Klaster 1. Ini bisa menjadi segmen **properti menengah atas atau premium**.
        """)

    st.subheader("4. Visualisasi Harga per Klaster")
    fig_boxplot, ax_boxplot = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='cluster', y='price_in_rp', data=df_clustered, palette='viridis', ax=ax_boxplot)
    ax_boxplot.set_title('Distribusi Harga Rumah per Klaster')
    ax_boxplot.set_xlabel('Klaster')
    ax_boxplot.set_ylabel('Harga (Rp)')
    st.pyplot(fig_boxplot)
    st.write("Visualisasi ini dengan jelas menunjukkan perbedaan rentang harga antar klaster.")

except FileNotFoundError:
    st.error("Pastikan 'jabodetabek_house_price.csv', 'preprocessor.pkl', dan 'kmeans_cluster_model.pkl' ada di direktori yang sama untuk menampilkan analisis klastering.")
    st.warning("Anda dapat menampilkan ringkasan klaster secara statis dari hasil notebook Anda di sini.")
    st.subheader("Ringkasan Klaster (Statistik dari Notebook)")
    st.markdown("""
        * **Klaster 0 (Harga Rendah - Menengah)**:
            * Jumlah: 2294
            * Harga Rata-rata: Rp. 1.37 Miliar
            * Luas Tanah Rata-rata: 104 m2
            * Luas Bangunan Rata-rata: 94 m2
            * Karakteristik: Properti yang lebih kecil dan terjangkau.
        * **Klaster 1 (Harga Sangat Tinggi - Anomali)**:
            * Jumlah: 4
            * Harga Rata-rata: Rp. 31.2 Miliar
            * Luas Tanah Rata-rata: 762 m2
            * Luas Bangunan Rata-rata: 2929 m2
            * Karakteristik: Properti yang sangat besar dan sangat mahal, kemungkinan data outlier.
        * **Klaster 2 (Harga Menengah - Tinggi)**:
            * Jumlah: 1255
            * Harga Rata-rata: Rp. 9.25 Miliar
            * Luas Tanah Rata-rata: 386 m2
            * Luas Bangunan Rata-rata: 345 m2
            * Karakteristik: Properti yang lebih besar dan mewah.
        """)