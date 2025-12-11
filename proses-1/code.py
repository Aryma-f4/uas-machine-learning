# Langkah-langkah Pemrosesan Data dan Klasifikasi
# 1. Cek Missing Value
# 2. Transformasi (MinMaxScaler)
# 3. Ekstraksi Fitur (LDA)
# 4. Imbalanced Data (ROS)
# 5. Klasifikasi (Random Forest)
# 6. Evaluasi Model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Abaikan peringatan untuk kejelasan output
warnings.filterwarnings("ignore")

# Pastikan Anda telah menginstal pustaka yang diperlukan:
# pip install pandas scikit-learn imbalanced-learn matplotlib seaborn

# --- 1. Memuat Dataset ---
# Menggunakan nama file yang diunggah
file_path = '../datasets/heart.csv'
try:
    df = pd.read_csv(file_path)
    print("Dataset berhasil dimuat.")
except FileNotFoundError:
    print(f"Error: File '{file_path}' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    exit()

# Pisahkan fitur (X) dan target (y)
X = df.drop('target', axis=1)
y = df['target']

# --- 1. Cek Missing Value ---
print("\n--- 1. Pengecekan Missing Value ---")
missing_values = df.isnull().sum()
print("Jumlah Missing Value per Kolom:")
print(missing_values[missing_values > 0])
if missing_values.sum() == 0:
    print("Tidak ada missing value (NaN) dalam dataset.")

# --- 2. Transformasi: MinMaxScaler ---
# Normalisasi data untuk memastikan semua fitur berada dalam rentang [0, 1]
print("\n--- 2. Transformasi Data: MinMaxScaler ---")
scaler = MinMaxScaler()
# Terapkan scaler pada seluruh fitur X
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print(f"Data fitur berhasil dinormalisasi menggunakan MinMaxScaler. Bentuk data: {X_scaled.shape}")

# --- Pembagian Data (Sebelum ROS dan LDA) ---
# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\nData dibagi. Ukuran Training: {X_train.shape}, Ukuran Testing: {X_test.shape}")
print(f"Distribusi kelas awal di data training:\n{y_train.value_counts()}")

# --- 4. Imbalanced Data: Random Over Sampling (ROS) ---
# Terapkan ROS hanya pada data pelatihan untuk menyeimbangkan kelas
print("\n--- 4. Penanganan Imbalanced Data: Random Over Sampling (ROS) ---")
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

print(f"Data training setelah ROS. Bentuk baru: {X_train_res.shape}")
print(f"Distribusi kelas baru di data training:\n{y_train_res.value_counts()}")

# --- 3. Ekstraksi Fitur: LDA (Linear Discriminant Analysis) ---
# Terapkan LDA hanya pada data training yang sudah di-resample (fit) dan transformasikan data test
# Jumlah komponen LDA akan menjadi min(n_kelas - 1, n_fitur) = min(2 - 1, 13) = 1
print("\n--- 3. Ekstraksi Fitur: Linear Discriminant Analysis (LDA) ---")
n_components = 1
lda = LDA(n_components=n_components)

# Fit LDA pada data training yang sudah di-resample
lda.fit(X_train_res, y_train_res)

# Transformasikan data training dan testing
X_train_lda = lda.transform(X_train_res)
X_test_lda = lda.transform(X_test)

print(f"Jumlah fitur setelah LDA: {X_train_lda.shape[1]} komponen.")
print(f"Data training setelah LDA. Bentuk: {X_train_lda.shape}")
print(f"Data testing setelah LDA. Bentuk: {X_test_lda.shape}")


# --- 5. Metode Klasifikasi: Random Forest Classifier ---
print("\n--- 5. Metode Klasifikasi: Random Forest Classifier ---")
# Inisialisasi model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Latih model menggunakan data training yang sudah di-resample dan di-LDA
model.fit(X_train_lda, y_train_res)
print("Model Random Forest berhasil dilatih.")

# Lakukan prediksi pada data testing yang sudah di-LDA
y_pred = model.predict(X_test_lda)


# --- 6. Evaluasi Model ---
print("\n--- 6. Evaluasi Model Klasifikasi ---")

# a. Akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model pada Data Testing: {accuracy:.4f}")

# b. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# c. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Visualisasi Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Prediksi 0', 'Prediksi 1'], 
            yticklabels=['Aktual 0', 'Aktual 1'])
plt.title('Confusion Matrix (Random Forest + ROS + LDA)')
plt.ylabel('Nilai Aktual')
plt.xlabel('Nilai Prediksi')
plt.show()

print("\nProses analisis selesai.")
