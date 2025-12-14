# Langkah-langkah Proses 3 (Daniel)
# 1. Cek Missing Value
# 2. Transformasi (MinMaxScaler)
# 3. Ekstraksi Fitur (LDA)
# 4. Klasifikasi (MLP Neural Network)
# 5. Evaluasi Model (Akurasi, Recall, Presisi, AUC/ROC, Waktu)

import pandas as pd
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, recall_score, precision_score

# Abaikan peringatan
warnings.filterwarnings("ignore")

# --- 1. Memuat Dataset ---
file_path = '../datasets/heart.csv' 
try:
    df = pd.read_csv(file_path)
    print("Dataset berhasil dimuat.")
except FileNotFoundError:
    print(f"Error: File '{file_path}' tidak ditemukan.")
    exit()

# Pisahkan fitur (X) dan target (y)
X = df.drop('target', axis=1)
y = df['target']

# --- Cek Missing Value ---
print("\n--- 1. Pengecekan Missing Value ---")
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print(missing_values[missing_values > 0])
    df = df.dropna()
    print("Missing value didrop.")
else:
    print("Tidak ada missing value.")

# --- 2. Transformasi: MinMaxScaler ---
print("\n--- 2. Transformasi Data: MinMaxScaler ---")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print(f"Data dinormalisasi. Bentuk: {X_scaled.shape}")

# --- Pembagian Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Data dibagi. Train: {X_train.shape}, Test: {X_test.shape}")

# --- 3. Ekstraksi Fitur: LDA ---
print("\n--- 3. Ekstraksi Fitur: Linear Discriminant Analysis (LDA) ---")
# n_components = 1 (Karena output binary class)
lda = LDA(n_components=1)

# Fit LDA pada data training
lda.fit(X_train, y_train)

# Transformasi X_train dan X_test
X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)
print(f"Bentuk Data setelah LDA (Train): {X_train_lda.shape}")

# --- 4. Metode Klasifikasi: MLP Neural Network ---
print("\n--- 4. Metode Klasifikasi: MLP Neural Network ---")
# Konfigurasi Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, activation='relu', solver='adam', random_state=42)

# Training (Hitung Waktu)
start_train = time.time()
mlp.fit(X_train_lda, y_train)
end_train = time.time()
train_time = end_train - start_train
print(f"Model MLP berhasil dilatih.")

# Testing (Hitung Waktu)
start_test = time.time()
y_pred = mlp.predict(X_test_lda)
y_proba = mlp.predict_proba(X_test_lda)[:, 1] # Probabilitas untuk AUC-ROC
end_test = time.time()
test_time = end_test - start_test

# --- 5. Evaluasi Model ---
print("\n--- 5. Evaluasi Model Klasifikasi ---")

print(f"Waktu Training : {train_time:.4f} detik")
print(f"Waktu Testing  : {test_time:.4f} detik")
print("-" * 30)
print(f"Akurasi        : {accuracy_score(y_test, y_pred):.4f}")
print(f"Presisi (Macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall (Macro) : {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"AUC-ROC        : {roc_auc_score(y_test, y_proba):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualisasi & Simpan Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Prediksi 0', 'Prediksi 1'],
            yticklabels=['Aktual 0', 'Aktual 1'])
plt.title('Confusion Matrix (MLP + LDA)')
plt.ylabel('Aktual')
plt.xlabel('Prediksi')
plt.tight_layout()

# Simpan gambar ke file
nama_file_gambar = 'confusion_matrix_proses3.png'
plt.savefig(nama_file_gambar) 
print(f"\nConfusion Matrix telah disimpan sebagai '{nama_file_gambar}'")