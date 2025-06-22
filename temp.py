
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load preprocessor and model
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('best_model.pkl')

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
    # Add dummy columns for features expected by preprocessor but not used in input
    # (url, title, address, facilities, ads_id, certificate, electricity)
    for col in ['url', 'title', 'address', 'facilities', 'ads_id', 'certificate', 'electricity']:
        input_df[col] = 'unknown'
    # Reorder columns to match training data
    ordered_cols = [
        'url', 'title', 'address', 'district', 'city', 'lat', 'long', 'facilities',
        'property_type', 'ads_id', 'bedrooms', 'bathrooms', 'land_size_m2', 'building_size_m2',
        'carports', 'certificate', 'electricity', 'maid_bedrooms', 'maid_bathrooms', 'floors',
        'building_age', 'year_built', 'property_condition', 'building_orientation', 'garages',
        'furnishing', 'price_per_m2', 'total_rooms', 'house_age_category'
    ]
    input_df = input_df[ordered_cols]
    processed_input = preprocessor.transform(input_df)
    predicted_price = model.predict(processed_input)[0]
    return predicted_price

def main():
    st.set_page_config(page_title="Prediksi Harga Rumah Jabodetabek", layout="wide")
    st.markdown("<h1 style='color:#2E86C1;'>Aplikasi Prediksi Harga Rumah Jabodetabek</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:18px;'>
        Masukkan detail properti rumah di bawah ini untuk memprediksi harga rumah di wilayah Jabodetabek menggunakan model machine learning terbaik.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Static metrics (from notebook, Gradient Boosting)
    st.info("**Performa Model (Gradient Boosting):**  \n"
            "RMSE: Rp. 3.077.916.144,00  \n"
            "MAE: Rp. 750.686.158,00  \n"
            "R2 Score: 0.8701")

    st.subheader("Prediksi Harga Rumah")

    # Example options for selectbox (from dataset, can be expanded)
    district_options = [
        "Summarecon Bekasi", "Mustikajaya", "Pondok Ungu", "Pondok Indah", "Ciparigi", "Parung", "Sentul City",
        "Sutera Onix Alam Sutera", "Sindang Jaya", "Jombang", "Lengkong Kulon", "BSD Provance Parkland", "Sudimara"
    ]
    city_options = [
        "Bekasi", "Tangerang", "Jakarta Selatan", "Bogor"
    ]
    property_type_options = [
        "rumah"
    ]
    property_condition_options = [
        "bagus", "bagus sekali", "baru"
    ]
    building_orientation_options = [
        "selatan", "utara", "timur", "barat"
    ]
    furnishing_options = [
        "unfurnished", "semi furnished", "furnished"
    ]
    house_age_category_options = [
        "baru", "sedang", "lama"
    ]

    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=-6.2, format="%.6f")
        long = st.number_input("Longitude", value=106.8, format="%.6f")
        bedrooms = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Jumlah Kamar Mandi", min_value=1, max_value=10, value=2)
        land_size_m2 = st.number_input("Luas Tanah (m2)", min_value=10.0, max_value=2000.0, value=100.0)
        building_size_m2 = st.number_input("Luas Bangunan (m2)", min_value=10.0, max_value=2000.0, value=80.0)
        carports = st.number_input("Jumlah Carport", min_value=0, max_value=5, value=1)
        maid_bedrooms = st.number_input("Jumlah Kamar ART", min_value=0, max_value=5, value=0)
    with col2:
        maid_bathrooms = st.number_input("Jumlah KM ART", min_value=0, max_value=5, value=0)
        floors = st.number_input("Jumlah Lantai", min_value=1, max_value=5, value=2)
        building_age = st.number_input("Usia Bangunan (tahun)", min_value=0, max_value=100, value=1)
        year_built = st.number_input("Tahun Dibangun", min_value=1900, max_value=2100, value=2021)
        garages = st.number_input("Jumlah Garasi", min_value=0, max_value=5, value=0)
        price_per_m2 = st.number_input("Harga per m2", min_value=1000000.0, max_value=1e8, value=1e7, step=1e6, format="%.0f")
        total_rooms = st.number_input("Total Ruangan", min_value=1, max_value=30, value=5)

    st.markdown("### Fitur Kategorikal")
    col3, col4, col5 = st.columns(3)
    with col3:
        district = st.selectbox("Kecamatan/Distrik", district_options)
        city = st.selectbox("Kota", city_options)
    with col4:
        property_type = st.selectbox("Tipe Properti", property_type_options)
        property_condition = st.selectbox("Kondisi Properti", property_condition_options)
    with col5:
        building_orientation = st.selectbox("Arah Bangunan", building_orientation_options)
        furnishing = st.selectbox("Furnishing", furnishing_options)
        house_age_category = st.selectbox("Kategori Usia Rumah", house_age_category_options)

    if st.button("Prediksi Harga Rumah"):
        pred = predict_house_price(
            lat, long, bedrooms, bathrooms, land_size_m2, building_size_m2, carports,
            maid_bedrooms, maid_bathrooms, floors, building_age, year_built, garages,
            price_per_m2, total_rooms,
            district, city, property_type, property_condition, building_orientation, furnishing, house_age_category
        )
        st.success(f"Prediksi Harga Rumah: {format_rupiah(pred)}")

if __name__ == '__main__':
    main()
