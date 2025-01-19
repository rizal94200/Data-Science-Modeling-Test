import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# Memuat model yang telah dilatih
with open('models.pkl', 'rb') as f:
    models = pickle.load(f)

# Judul Aplikasi
st.title("Prediksi Supply Chain Management")

# Deskripsi Aplikasi
st.write("""
Aplikasi ini menggunakan model Random Forest Regressor untuk memprediksi berbagai nilai terkait 
dengan supply chain management berdasarkan dua fitur: `storageCost` dan `interestRate`.
""")

# Input dari pengguna untuk fitur
storage_cost = st.number_input("Masukkan nilai Storage Cost", min_value=0.0, value=0.0, step=0.01)
interest_rate = st.number_input("Masukkan nilai Interest Rate", min_value=0.0, value=0.0, step=0.01)

# Mengonversi input fitur menjadi DataFrame
input_features = pd.DataFrame({
    'storageCost': [storage_cost],
    'interestRate': [interest_rate]
})

# Daftar kolom target yang akan diprediksi
target_columns = [
    'compidx0lt2', 'compidx0lt2l1', 'compidx0lt2l2', 'compidx0lt2l4', 'compidx0lt2l8', 
    'compidx1lt2', 'compidx2lt2', 'compidx3lt2', 'compidx4lt2', 'compidx4lt2l1', 
    'compidx4lt2l2', 'compidx4lt2l4', 'compidx5lt2', 'compidx6lt2', 'compidx7lt2', 
    'compidx8lt2'
]

# Tombol untuk melakukan prediksi
if st.button("Prediksi"):
    st.write("Menjalankan prediksi...")

    # Menyimpan hasil prediksi dalam dictionary
    predictions = {}

    # Melakukan prediksi untuk setiap target dan menampilkan hasilnya
    for target_column in target_columns:
        model = models[target_column]
        prediction = model.predict(input_features)
        
        predictions[target_column] = prediction[0]
        
        st.write(f"**Prediksi untuk {target_column}:**")
        st.write(f"  Prediksi nilai: {prediction[0]:.2f}")

    # Analisis Prediksi
    st.subheader("Analisis Prediksi")

    # Menampilkan grafik perbandingan antara nilai prediksi dan input fitur
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(predictions.keys(), predictions.values(), color='skyblue')
    ax.set_xlabel('Target Columns')
    ax.set_ylabel('Prediksi Nilai')
    ax.set_title('Perbandingan Prediksi untuk Setiap Target')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # Menghitung dan menampilkan nilai rata-rata prediksi
    avg_prediction = np.mean(list(predictions.values()))
    st.write(f"**Nilai rata-rata dari semua prediksi:** {avg_prediction:.2f}")
