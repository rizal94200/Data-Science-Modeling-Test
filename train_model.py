import pandas as pd
import arff
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Path ke dataset
data_train_path = '/content/Supply Chain Management_train.arff'
data_test_path = '/content/Supply Chain Management_test.arff'


# Baca dataset ARFF dan konversi ke DataFrame
with open(data_train_path, 'r') as f:
    train_data = arff.load(f)
df_train = pd.DataFrame(list(train_data['data']), columns=[attr[0] for attr in train_data['attributes']])

with open(data_test_path, 'r') as f:
    test_data = arff.load(f)
df_test = pd.DataFrame(list(test_data['data']), columns=[attr[0] for attr in test_data['attributes']])

# Menghapus nilai yang hilang
df_train = df_train.dropna()
df_test = df_test.dropna()

# EDA - Statistik Deskriptif
print("**Statistik Deskriptif Data Latih:**")
print(df_train.describe())

# EDA - Cek Distribusi Kolom Fitur
plt.figure(figsize=(10, 6))
sns.histplot(df_train['storageCost'], kde=True)
plt.title('Distribusi Storage Cost')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df_train['interestRate'], kde=True)
plt.title('Distribusi Interest Rate')
plt.show()

# EDA - Korelasi antar fitur
correlation_matrix = df_train[['storageCost', 'interestRate']].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap Korelasi Fitur')
plt.show()

# EDA - Cek distribusi target yang akan diprediksi
plt.figure(figsize=(10, 6))
sns.histplot(df_train['compidx0lt2'], kde=True)
plt.title('Distribusi Target compidx0lt2')
plt.show()

# Fitur yang akan digunakan
X_train = df_train[['storageCost', 'interestRate']]
X_test = df_test[['storageCost', 'interestRate']]

# Daftar kolom target yang akan diprediksi
target_columns = [
    'compidx0lt2', 'compidx0lt2l1', 'compidx0lt2l2', 'compidx0lt2l4', 'compidx0lt2l8', 
    'compidx1lt2', 'compidx2lt2', 'compidx3lt2', 'compidx4lt2', 'compidx4lt2l1', 
    'compidx4lt2l2', 'compidx4lt2l4', 'compidx5lt2', 'compidx6lt2', 'compidx7lt2', 
    'compidx8lt2'
]

# Inisialisasi dictionary untuk menyimpan hasil model
models = {}

# Latih dan evaluasi model untuk setiap target
for target_column in target_columns:
    y_train = df_train[target_column]
    y_test = df_test[target_column]
    
    # Buat model Random Forest Regressor (untuk prediksi kontinu)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Latih model
    model.fit(X_train, y_train)
    
    # Simpan model ke dalam dictionary
    models[target_column] = model
    
    # Prediksi untuk data uji
    y_pred = model.predict(X_test)
    
    # Evaluasi model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Evaluasi Model untuk {target_column}:")
    print(f"  Mean Squared Error: {mse}")
    print(f"  R-squared: {r2}")
    print("\n")

# Simpan semua model terlatih ke file
with open('models.pkl', 'wb') as f:
    pickle.dump(models, f)
