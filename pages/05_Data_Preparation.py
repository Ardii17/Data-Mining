# pages/05_Data_Preparation.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Data Preparation",
    layout="wide"
)

st.markdown("<h1 style='color:#2E86C1;'>Data Preparation</h1>", unsafe_allow_html=True)
st.markdown("---")

st.write("Tahap ini melibatkan pembersihan data, penanganan *missing values*, rekayasa fitur (*feature engineering*), dan persiapan data untuk pemodelan.")

st.subheader("1. Penanganan Missing Values")
st.write("""
Nilai-nilai yang hilang (*missing values*) dalam dataset ditangani untuk memastikan kualitas dan kelengkapan data.
Pada kasus ini, *missing values* pada kolom numerik diisi dengan **median** dari kolom tersebut,
sementara *missing values* pada kolom kategorikal diisi dengan **modus** (nilai yang paling sering muncul) atau 'Unknown' jika tidak ada modus.
""")

# Load data to show effect of preprocessing
@st.cache_data
def get_processed_df(file_path):
    df_prep = pd.read_csv(file_path)
    # Replicate preprocessing from notebook cell 10 & 11
    num_cols_prep = df_prep.select_dtypes(include=[np.number]).columns
    df_prep[num_cols_prep] = df_prep[num_cols_prep].apply(lambda x: x.fillna(x.median()))
    cat_cols_prep = df_prep.select_dtypes(include=['object']).columns
    df_prep[cat_cols_prep] = df_prep[cat_cols_prep].apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))

    df_prep['price_per_m2'] = df_prep['price_in_rp'] / df_prep['land_size_m2']
    df_prep['total_rooms'] = df_prep['bedrooms'] + df_prep['bathrooms'] + df_prep['maid_bedrooms'] + df_prep['maid_bathrooms']
    
    def age_category(age):
        if pd.isnull(age):
            return 'unknown'
        elif age <= 2:
            return 'baru'
        elif age <= 10:
            return 'sedang'
        else:
            return 'lama'
    df_prep['house_age_category'] = df_prep['building_age'].apply(age_category)
    return df_prep

df_prepared = get_processed_df('jabodetabek_house_price.csv')

if df_prepared is not None:
    st.write("Setelah penanganan *missing values*, tidak ada lagi nilai yang hilang:")
    st.dataframe(df_prepared.isnull().sum().rename('Missing Values Setelah Penanganan'))

st.subheader("2. Feature Engineering")
st.write("""
Beberapa fitur baru direkayasa untuk meningkatkan informasi yang dapat ditangkap oleh model:
-   **`price_per_m2`**: Dihitung dari `price_in_rp` dibagi `land_size_m2`. Fitur ini memberikan gambaran tentang kepadatan harga per meter persegi.
-   **`total_rooms`**: Merupakan jumlah total `bedrooms`, `bathrooms`, `maid_bedrooms`, dan `maid_bathrooms`. Fitur ini merepresentasikan total ruang fungsional dalam properti.
-   **`house_age_category`**: Kategori usia bangunan (`building_age`) diubah menjadi 'baru' (usia <= 2 tahun), 'sedang' (usia > 2 dan <= 10 tahun), atau 'lama' (usia > 10 tahun).
""")
st.write("Contoh fitur hasil rekayasa:")
st.dataframe(df_prepared[['price_per_m2', 'total_rooms', 'house_age_category']].head())

st.subheader("3. Pemisahan Fitur dan Target")
st.write("""
Dataset dipisahkan menjadi fitur (`X`) dan variabel target (`y`).
-   **Variabel Target (`y`)**: `price_in_rp` (harga rumah dalam Rupiah).
-   **Fitur (`X`)**: Semua kolom lainnya yang relevan setelah pra-pemrosesan.
""")
st.write(f"Bentuk data fitur (X): {df_prepared.drop('price_in_rp', axis=1).shape}")
st.write(f"Bentuk data target (y): {df_prepared['price_in_rp'].shape}")

st.subheader("4. Pembagian Data Training dan Testing")
st.write("""
Data kemudian dibagi menjadi set data *training* dan *testing* dengan rasio 80% (training) dan 20% (testing).
Pembagian ini penting untuk melatih model pada satu set data dan mengevaluasinya pada set data yang belum pernah dilihat model sebelumnya,
untuk mengukur kemampuan generalisasi model.
""")
# Statistik dari notebook cell 14
st.write(f"Jumlah data training: **2842**")
st.write(f"Jumlah data testing: **711**")

st.subheader("5. Preprocessing Pipeline (Scaling dan Encoding)")
st.write("""
Pipeline pra-pemrosesan (`ColumnTransformer`) digunakan untuk mengotomatisasi transformasi fitur:
-   **Fitur Numerik**: Diskala menggunakan `StandardScaler` untuk menstandarkan rentang nilai.
-   **Fitur Kategorikal**: Di-*encode* menggunakan `OneHotEncoder` untuk mengubahnya menjadi format numerik yang dapat dipahami model. `handle_unknown='ignore'` digunakan untuk menangani kategori baru yang mungkin muncul di data *testing*.
""")
st.write("Proses *fit_transform* dilakukan pada data *training* dan *transform* pada data *testing*.")
# Statistik dari notebook cell 15
st.write("Shape X_train_processed: (2842, 10811)")
st.write("Shape X_test_processed: (711, 10811)")