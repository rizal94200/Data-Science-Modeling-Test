#!/bin/bash

# Jalankan perintah untuk melatih model
echo "Melatih model..."
python train_model.py

# Periksa apakah perintah sebelumnya berhasil
if [ $? -ne 0 ]; then
  echo "Gagal melatih model. Menjalankan aplikasi Streamlit secara langsung..."
  streamlit run app.py
else
  echo "Model berhasil dilatih. Menjalankan aplikasi Streamlit..."
  streamlit run app.py
fi