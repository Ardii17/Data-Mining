# pages/07_House_Price_Prediction.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


st.set_page_config(
    page_title="Prediksi Harga Rumah",
    layout="wide"
)

# Load preprocessor and model
try:
    preprocessor = joblib.load('preprocessor.pkl')
    model = joblib.load('best_model.pkl')
except FileNotFoundError:
    st.error("File model atau preprocessor tidak ditemukan. Pastikan 'preprocessor.pkl' dan 'best_model.pkl' ada di direktori yang sama.")
    st.stop() # Hentikan eksekusi jika file tidak ditemukan

# Helper for currency formatting
def format_rupiah(value):
    return f"Rp. {value:,.2f}".replace(",", ".").replace(".", ",", 1)

# Prediction function
def predict_house_price(
    lat, long, bedrooms, bathrooms, land_size_m2, building_size_m2, carports,
    maid_bedrooms, maid_bathrooms, floors, building_age, year_built, garages,
    price_per_m2, total_rooms,
    district, city, property_type, property_condition, building_orientation, furnishing, house_age_category
):
    # Compose input as DataFrame with correct column order
    input_dict = {
        'lat': [lat],
        'long': [long],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'land_size_m2': [land_size_m2],
        'building_size_m2': [building_size_m2],
        'carports': [carports],
        'maid_bedrooms': [maid_bedrooms],
        'maid_bathrooms': [maid_bathrooms],
        'floors': [floors],
        'building_age': [building_age],
        'year_built': [year_built],
        'garages': [garages],
        'price_per_m2': [price_per_m2],
        'total_rooms': [total_rooms],
        'district': [district],
        'city': [city],
        'property_type': [property_type],
        'property_condition': [property_condition],
        'building_orientation': [building_orientation],
        'furnishing': [furnishing],
        'house_age_category': [house_age_category]
    }
    input_df = pd.DataFrame(input_dict)
    
    # Get the exact column order from the preprocessor's numerical and categorical features
    # This requires knowing the original num_features and cat_features lists from your notebook.
    # A more robust way would be to save these lists alongside the preprocessor/model.
    # For now, let's explicitly list them as they were in your notebook's X.columns.
    # From tubes.ipynb cell 13:
    all_X_columns_template = [
        'url', 'title', 'address', 'district', 'city', 'lat', 'long', 'facilities',
        'property_type', 'ads_id', 'bedrooms', 'bathrooms', 'land_size_m2', 'building_size_m2',
        'carports', 'certificate', 'electricity', 'maid_bedrooms', 'maid_bathrooms', 'floors',
        'building_age', 'year_built', 'property_condition', 'building_orientation', 'garages',
        'furnishing', 'price_per_m2', 'total_rooms', 'house_age_category'
    ]

    # Create a dummy DataFrame with all expected columns and fill with placeholders
    # then update with user inputs. This ensures all columns are present and in order.
    # Default values for columns not directly taken as input:
    default_vals = {
        'url': 'http://dummy.url/',
        'title': 'Dummy Title',
        'address': 'Dummy Address',
        'facilities': 'None', # Or a common value from your data
        'ads_id': 'dummy_id',
        'certificate': 'shm - sertifikat hak milik', # Common from your data
        'electricity': '2200 mah' # Common from your data
    }
    
    # Create an empty DataFrame with the correct column order
    full_input_df = pd.DataFrame(columns=all_X_columns_template)
    
    # Create a single row dictionary from user inputs + defaults
    single_row_data_for_full_df = {}
    for col in all_X_columns_template:
        if col in input_dict:
            single_row_data_for_full_df[col] = input_dict[col][0]
        else:
            single_row_data_for_full_df[col] = default_vals.get(col, None) # Use .get() with None default for safety

    # Append the single row to the full_input_df
    # Using concat for newer pandas versions is common
    full_input_df = pd.concat([full_input_df, pd.DataFrame([single_row_data_for_full_df])], ignore_index=True)


    processed_input = preprocessor.transform(full_input_df)
    predicted_price = model.predict(processed_input)[0]
    return predicted_price

# --- Streamlit UI ---
st.markdown("<h1 style='color:#2E86C1;'>Prediksi Harga Rumah</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='font-size:18px;'>
        Gunakan formulir di bawah ini untuk memprediksi harga rumah di wilayah Jabodetabek.
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

# Static metrics (from notebook, Gradient Boosting)
st.info("**Performa Model Terbaik (Gradient Boosting):** \n"
          "RMSE: Rp. 3.077.916.144,00  \n" # Dari tubes.ipynb cell 16 output
          "MAE: Rp. 750.686.158,00  \n" # Dari tubes.ipynb cell 16 output
          "R2 Score: 0.8701") # Dari tubes.ipynb cell 16 output

st.subheader("Masukkan Detail Properti")

# Example options for selectbox (from dataset, can be expanded)
# These should ideally be obtained from unique values in your original dataset
# For demonstration, using a subset of common values.
# From df.describe(include='all') in tubes.ipynb:
district_options = [
    "Summarecon Bekasi", "Mustikajaya", "Pondok Ungu", "Pondok Indah", "Ciparigi", "Parung", "Sentul City",
    "Sutera Onix Alam Sutera", "Sindang Jaya", "Jombang", "Lengkong Kulon", "BSD Provance Parkland", "Sudimara",
    "Bogor", "Bekasi", "Tangerang", "Jakarta Selatan", "Kebon Jeruk", "Cilandak", "Kebayoran Baru",
    # Add more districts from your data if needed. You have 380 unique districts.
]
city_options = [
    "Bekasi", "Tangerang", "Jakarta Selatan", "Bogor", "Depok", "Jakarta Barat", "Jakarta Timur", "Jakarta Pusat", "Jakarta Utara" # All 9 cities from your df.describe
]
property_type_options = ["rumah"] # Your data implies only 'rumah' from df.describe
property_condition_options = [
    "bagus", "bagus sekali", "baru", "perlu renovasi", "sudah direnovasi", "standar", "jelek" # From df.describe
]
building_orientation_options = [
    "selatan", "utara", "timur", "barat", "tenggara", "barat laut", "timur laut", "barat daya" # From df.describe
]
furnishing_options = [
    "unfurnished", "semi furnished", "furnished", "kosong" # From df.describe (4 unique values)
]
house_age_category_options = [
    "baru", "sedang", "lama", "unknown" # From your feature engineering function
]


col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude", value=-6.223945, format="%.6f", help="Latitude lokasi properti. Contoh: -6.223945")
    long = st.number_input("Longitude", value=106.986275, format="%.6f", help="Longitude lokasi properti. Contoh: 106.986275")
    bedrooms = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=99, value=3, help="Jumlah kamar tidur di properti. (Max dari data Anda 99.0)") # Max value from df.describe: 99.0
    bathrooms = st.number_input("Jumlah Kamar Mandi", min_value=1, max_value=99, value=2, help="Jumlah kamar mandi di properti. (Max dari data Anda 99.0)") # Max value from df.describe: 99.0
    land_size_m2 = st.number_input("Luas Tanah (m2)", min_value=12.0, max_value=8000.0, value=100.0, help="Luas total tanah dalam meter persegi (m2). (Min 12.0, Max 8000.0)") # Min/Max from df.describe
    building_size_m2 = st.number_input("Luas Bangunan (m2)", min_value=1.0, max_value=6000.0, value=80.0, help="Luas total bangunan dalam meter persegi (m2). (Min 1.0, Max 6000.0)") # Min/Max from df.describe
    carports = st.number_input("Jumlah Carport", min_value=0, max_value=15, value=1, help="Jumlah carport yang tersedia. (Max dari data Anda 15.0)") # Max value from df.describe: 15.0
    maid_bedrooms = st.number_input("Jumlah Kamar ART", min_value=0, max_value=7, value=0, help="Jumlah kamar tidur untuk asisten rumah tangga. (Max dari data Anda 7.0)") # Max value from df.describe: 7.0
with col2:
    maid_bathrooms = st.number_input("Jumlah KM ART", min_value=0, max_value=5, value=0, help="Jumlah kamar mandi untuk asisten rumah tangga. (Max dari data Anda 5.0)") # Max value from df.describe: 5.0
    floors = st.number_input("Jumlah Lantai", min_value=1, max_value=5, value=2, help="Jumlah total lantai properti. (Max dari data Anda 5.0)") # Max value from df.describe: 5.0
    building_age = st.number_input("Usia Bangunan (tahun)", min_value=0, max_value=152, value=5, help="Usia bangunan dalam tahun. 0 untuk bangunan baru. (Max dari data Anda 152.0)") # Max value from df.describe: 152.0
    year_built = st.number_input("Tahun Dibangun", min_value=1870, max_value=2052, value=2020, help="Tahun bangunan selesai dibangun. (Min 1870.0, Max 2052.0)") # Min/Max from df.describe
    garages = st.number_input("Jumlah Garasi", min_value=0, max_value=50, value=0, help="Jumlah garasi yang tersedia. (Max dari data Anda 50.0)") # Max value from df.describe: 50.0
    # price_per_m2 dan total_rooms adalah fitur hasil rekayasa, pastikan nilainya realistis dengan input lainnya
    price_per_m2 = st.number_input("Harga per m2 (Rp)", min_value=10000.0, max_value=1.0e9, value=1.0e7, step=1000000.0, format="%.0f", help="Estimasi harga properti per meter persegi. Ini adalah fitur hasil rekayasa yang sangat berpengaruh.")
    total_rooms = st.number_input("Total Ruangan", min_value=1, max_value=150, value=5, help="Total kombinasi kamar tidur, kamar mandi, dan kamar ART. Ini adalah fitur hasil rekayasa.")

st.markdown("### Fitur Kategorikal")
col3, col4, col5 = st.columns(3)
with col3:
    district = st.selectbox("Kecamatan/Distrik", district_options, help="Kecamatan atau distrik lokasi properti.")
    city = st.selectbox("Kota", city_options, help="Kota lokasi properti.")
with col4:
    property_type = st.selectbox("Tipe Properti", property_type_options, help="Jenis properti (misalnya 'rumah').")
    property_condition = st.selectbox("Kondisi Properti", property_condition_options, help="Kondisi umum properti.")
with col5:
    building_orientation = st.selectbox("Arah Bangunan", building_orientation_options, help="Arah hadap utama bangunan.")
    furnishing = st.selectbox("Furnishing", furnishing_options, help="Status perabotan properti.")
    house_age_category = st.selectbox("Kategori Usia Rumah", house_age_category_options, help="Kategori usia bangunan: baru, sedang, atau lama.")

st.markdown("---")
if st.button("Prediksi Harga Rumah"):
    with st.spinner('Memprediksi harga...'):
        try:
            pred = predict_house_price(
                lat, long, bedrooms, bathrooms, land_size_m2, building_size_m2, carports,
                maid_bedrooms, maid_bathrooms, floors, building_age, year_built, garages,
                price_per_m2, total_rooms,
                district, city, property_type, property_condition, building_orientation, furnishing, house_age_category
            )
            st.success(f"Prediksi Harga Rumah: {format_rupiah(pred)}")
            st.balloons()
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memprediksi: {e}. Pastikan semua input valid.")
            st.warning("Periksa kembali input Anda, terutama untuk 'price_per_m2' dan 'total_rooms' karena ini adalah fitur hasil rekayasa.")