import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve

# 1. Load Data & Cek Missing Value
datasets = '.\datasets\heart.csv'
df = pd.read_csv(datasets)

print("--- 1. Cek Missing Value ---")
missing_values = df.isnull().sum()
print(missing_values)

# Persiapan Data
X = df.drop('target', axis=1)  # Fitur
y = df['target']              # Label

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 2. Transformasi --> MinMaxScaler
print("\n--- 2. Transformasi (MinMaxScaler) ---")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data scaling selesai.")

# 3. Metode klasifikasi = MLP Neural Network
print("\n--- 3. Training MLP Neural Network ---")
mlp = MLPClassifier(random_state=42, max_iter=1000)

# Hitung Waktu Training
start_train = time.time()
mlp.fit(X_train_scaled, y_train)
end_train = time.time()
training_time = end_train - start_train

# 4. Evaluasi
print("\n--- 4. Evaluasi ---")

# Hitung Waktu Testing (Prediksi)
start_test = time.time()
y_pred = mlp.predict(X_test_scaled)
end_test = time.time()
testing_time = end_test - start_test

# Hitung Metrik Evaluasi
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# Untuk AUC/ROC, gunakan probabilitas
y_prob = mlp.predict_proba(X_test_scaled)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)

# Print Hasil
print(f"Waktu Training: {training_time:.4f} detik")
print(f"Waktu Testing : {testing_time:.4f} detik")
print(f"Akurasi      : {accuracy:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"Presisi      : {precision:.4f}")
print(f"AUC Score    : {auc_score:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'MLP (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
