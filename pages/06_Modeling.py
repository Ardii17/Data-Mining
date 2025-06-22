# pages/06_Modeling.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # Only for displaying metrics

st.set_page_config(
    page_title="Modeling",
    layout="wide"
)

st.markdown("<h1 style='color:#2E86C1;'>Modeling</h1>", unsafe_allow_html=True)
st.markdown("---")

st.write("Pada tahap ini, berbagai model *supervised learning* (Regresi) dan *unsupervised learning* (Klastering) dibangun dan dievaluasi.")

st.subheader("1. Model Regresi (House Price Prediction)")
st.write("""
Beberapa algoritma regresi dilatih dan dievaluasi untuk memprediksi harga rumah.
Metrik evaluasi utama untuk regresi adalah:
-   **RMSE (Root Mean Squared Error)**: Mengukur rata-rata besarnya kesalahan prediksi. Semakin rendah, semakin baik.
-   **MAE (Mean Absolute Error)**: Mengukur rata-rata kesalahan absolut antara prediksi dan nilai aktual. Semakin rendah, semakin baik.
-   **R2 Score (Coefficient of Determination)**: Mengukur seberapa baik model menjelaskan variabilitas data target. Rentang 0 hingga 1, semakin mendekati 1 semakin baik.
""")

st.write("Berikut adalah ringkasan hasil evaluasi model regresi yang dilatih di *notebook*:")

# Data ini hardcoded dari output cell 16 di notebook Anda
results_data = {
    'Linear Regression': {'RMSE': 4.336412e+09, 'MAE': 1.990787e+09, 'R2': 0.742189},
    'Ridge Regression': {'RMSE': 4.327017e+09, 'MAE': 1.984019e+09, 'R2': 0.743305},
    'Lasso Regression': {'RMSE': 5.276377e+09, 'MAE': 2.825157e+09, 'R2': 0.618309},
    'Decision Tree': {'RMSE': 6.034670e+09, 'MAE': 1.013474e+09, 'R2': 0.500716},
    'Random Forest': {'RMSE': 3.947169e+09, 'MAE': 6.777262e+08, 'R2': 0.786395},
    'Gradient Boosting': {'RMSE': 3.077916e+09, 'MAE': 7.506862e+08, 'R2': 0.870117},
    'Support Vector Regressor': {'RMSE': 8.918639e+09, 'MAE': 3.231745e+09, 'R2': -0.090530}
}
results_df = pd.DataFrame(results_data).T
st.dataframe(results_df.style.format({"RMSE": "{:,.2f}", "MAE": "{:,.2f}", "R2": "{:.4f}"}))

st.markdown("""
Dari hasil di atas, **Gradient Boosting** menunjukkan performa terbaik dengan R2 Score tertinggi (0.8701) dan RMSE terendah.
Model ini kemudian dilakukan *hyperparameter tuning* untuk mendapatkan performa optimal.
""")
st.write("**Hasil Evaluasi Model Terbaik Setelah Tuning (Gradient Boosting):**")
st.write(f"- RMSE: Rp. 3.188.174.390,20") # From notebook cell 17 output
st.write(f"- MAE: Rp. 649.374.661,97") # From notebook cell 17 output
st.write(f"- R2 Score: 0.8606") # From notebook cell 17 output

st.subheader("2. Model Unsupervised Learning (Klastering)")
st.write("""
Untuk memahami segmentasi properti, model *unsupervised learning* **K-Means Clustering** digunakan.
Tahapan utamanya meliputi:
-   **Pra-pemrosesan Data**: Data disiapkan dan diskalakan agar cocok untuk algoritma klastering.
-   **Menentukan Jumlah Klaster Optimal (Elbow Method & Silhouette Score)**: Metode ini membantu menentukan berapa banyak klaster yang paling sesuai untuk data. Berdasarkan analisis di *notebook*, **jumlah klaster optimal yang dipilih adalah 3**.
-   **Melatih Model K-Means**: Model dilatih dengan jumlah klaster yang ditentukan.
-   **Analisis Hasil Klastering**: Karakteristik setiap klaster dianalisis berdasarkan fitur-fitur seperti harga rata-rata, luas tanah, jumlah kamar, dll.
""")
st.write("Detail analisis klastering dapat Anda lihat di halaman 'Analisis Klastering'.")