# pages/02_Business_Understanding.py
import streamlit as st

st.set_page_config(
    page_title="Business Understanding",
    layout="wide"
)

st.markdown("<h1 style='color:#2E86C1;'>Business Understanding</h1>", unsafe_allow_html=True)
st.markdown("---")

st.subheader("1. Objective")
st.write("""
Tujuan utama dari proyek ini adalah untuk membangun model *machine learning* yang mampu memprediksi harga rumah secara akurat di wilayah Jabodetabek.
Selain itu, proyek ini juga bertujuan untuk melakukan analisis klastering guna mengidentifikasi segmen-segmen properti yang berbeda berdasarkan karakteristiknya.
""")

st.subheader("2. Business Understanding")
st.write("""
Prediksi harga rumah memiliki nilai bisnis yang signifikan bagi berbagai pemangku kepentingan:

* **Bagi Pembeli Properti:** Membantu dalam memperkirakan nilai wajar properti, sehingga dapat membuat keputusan pembelian yang lebih informatif dan menghindari harga yang terlalu tinggi.
* **Bagi Penjual Properti:** Memungkinkan penetapan harga jual yang kompetitif dan realistis, mempercepat proses penjualan, dan memaksimalkan keuntungan.
* **Bagi Agen Properti:** Memberikan alat bantu untuk memberikan rekomendasi harga kepada klien secara lebih objektif dan cepat.
* **Bagi Investor Properti:** Membantu dalam mengidentifikasi peluang investasi yang menguntungkan dengan memprediksi potensi apresiasi atau depresiasi nilai properti.
* **Bagi Lembaga Keuangan (Bank/KPR):** Mempermudah proses penilaian agunan (properti) untuk pengajuan kredit, sehingga mengurangi risiko dan mempercepat persetujuan.

Dengan model prediksi yang akurat dan pemahaman klasterisasi pasar, berbagai pihak dapat mengambil keputusan yang lebih cerdas di pasar properti yang dinamis.
""")