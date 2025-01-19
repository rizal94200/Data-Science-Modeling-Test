import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Memuat model yang telah dilatih
with open('models.pkl', 'rb') as f:
    models = pickle.load(f)

# Judul Aplikasi
st.title("Prediksi Supply Chain Management")

# Deskripsi Aplikasi
st.write("""
Aplikasi ini menggunakan model Random Forest Regressor untuk memprediksi berbagai nilai terkait 
dengan supply chain management berdasarkan dua fitur: storageCost dan interestRate.
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
        
        st.write(f"*Prediksi untuk {target_column}:*")
        st.write(f"  Prediksi nilai: {prediction[0]:.2f}")

    # Analisis Prediksi
    st.subheader("Analisis Prediksi")

    # Menyimpan nilai maksimum, minimum, dan rata-rata
    max_prediction = max(predictions.values())
    min_prediction = min(predictions.values())
    avg_prediction = np.mean(list(predictions.values()))

    # Grafik nilai maksimum
    st.write(f"*Nilai maksimum dari semua prediksi:* {max_prediction:.2f}")
    fig_max, ax_max = plt.subplots(figsize=(12, 6))
    ax_max.bar(predictions.keys(), predictions.values(), color='skyblue')
    ax_max.axhline(y=max_prediction, color='green', linestyle='--', label=f'Maksimum ({max_prediction:.2f})')
    ax_max.set_xlabel('Target Columns')
    ax_max.set_ylabel('Prediksi Nilai')
    ax_max.set_title('Grafik Nilai Maksimum')
    plt.xticks(rotation=90)
    plt.legend()
    st.pyplot(fig_max)

    # Grafik nilai minimum
    st.write(f"*Nilai minimum dari semua prediksi:* {min_prediction:.2f}")
    fig_min, ax_min = plt.subplots(figsize=(12, 6))
    ax_min.bar(predictions.keys(), predictions.values(), color='skyblue')
    ax_min.axhline(y=min_prediction, color='red', linestyle='--', label=f'Minimum ({min_prediction:.2f})')
    ax_min.set_xlabel('Target Columns')
    ax_min.set_ylabel('Prediksi Nilai')
    ax_min.set_title('Grafik Nilai Minimum')
    plt.xticks(rotation=90)
    plt.legend()
    st.pyplot(fig_min)

    # Grafik nilai rata-rata
    st.write(f"*Nilai rata-rata dari semua prediksi:* {avg_prediction:.2f}")
    fig_avg, ax_avg = plt.subplots(figsize=(12, 6))
    ax_avg.bar(predictions.keys(), predictions.values(), color='skyblue')
    ax_avg.axhline(y=avg_prediction, color='orange', linestyle='--', label=f'Rata-rata ({avg_prediction:.2f})')
    ax_avg.set_xlabel('Target Columns')
    ax_avg.set_ylabel('Prediksi Nilai')
    ax_avg.set_title('Grafik Nilai Rata-rata')
    plt.xticks(rotation=90)
    plt.legend()
    st.pyplot(fig_avg)