# ===============================
# PROSES 4
# Cek Missing Value
# Transformasi MinMaxScaler
# SMOTE
# Klasifikasi (1 metode)
# Evaluasi
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE

# ===============================
# Load Dataset
# ===============================
data_path = "../datasets/heart.csv"
data = pd.read_csv(data_path)

print("Dataset berhasil dimuat")
print(data.head())

# ===============================
# 1. Cek Missing Value
# ===============================
print("\nMissing Value:")
print(data.isnull().sum())

# Jika ada missing value (opsional)
data = data.dropna()

# ===============================
# Pisahkan fitur dan target
# ===============================
X = data.drop("target", axis=1)
y = data["target"]

# ===============================
# Split Data
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 2. Transformasi (MinMaxScaler)
# ===============================
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\n data sebelum scaling:")
print(X_train.iloc[0])

print("\n data setelah MinMaxScaler:")
print(X_train_scaled[0])

# ===============================
# 3. SMOTE (Imbalanced Data)
# ===============================
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nDistribusi kelas setelah SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# ===============================
# 4. Metode Klasifikasi
# Logistic Regression
# ===============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_smote, y_train_smote)

print("\n=== MODEL KLASIFIKASI ===")
print("Metode       : Logistic Regression")
print("Jumlah data latih :", X_train_smote.shape[0])
print("Jumlah fitur     :", X_train_smote.shape[1])


# ===============================
# Prediksi
# ===============================
y_pred = model.predict(X_test_scaled)

# ===============================
# Evaluasi
# ===============================
print("\n=== HASIL EVALUASI ===")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# Visualisasi Confusion Matrix
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Simpan gambar
plt.savefig("Figure_4.png")
plt.show()
