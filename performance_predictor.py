"""
performance_predictor.py
VERSI FINAL:
- 3 MODEL: Linear Regression, RandomForest, XGBoost
- Database logging: model_results + prediction_history + data_info
- Penanganan SHAP modern (Explainer + Data Sample)
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import shap
import joblib
from datetime import datetime, UTC

# ML
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# =====================================================
# KONFIGURASI
# =====================================================
MODEL_DIR = "models"
DB_NAME = "student_performance.db"

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
SHAP_PATH = os.path.join(MODEL_DIR, "shap_explainer_package.joblib")
DATA_PATH = "StudentPerformanceFactors.csv"

FEATURE_COLUMNS = ["Sleep_Hours", "Hours_Studied"]
TARGET_COLUMN = "Exam_Score"

os.makedirs(MODEL_DIR, exist_ok=True)


# =====================================================
# DATABASE UTILITIES
# =====================================================
def get_db_connection():
    """Membuat koneksi ke database dan mendefinisikan tabel jika belum ada."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Tabel 1: Model Results
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_results (
        id INTEGER PRIMARY KEY,
        model_name TEXT NOT NULL UNIQUE,
        r2_score REAL,
        rmse_score REAL,
        is_best INTEGER
    )
    """)
    
    # Tabel 2: Prediction History
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY,
        sleep_hours REAL,
        studied_hours REAL,
        predicted_score REAL,
        timestamp TEXT
    )
    """)
    
    # Tabel 3: Data Train/Test Info
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS data_info (
        key TEXT PRIMARY KEY,
        value REAL
    )
    """)
    
    conn.commit()
    return conn

def save_model_result(conn, name, r2, rmse, is_best=0):
    """Menyimpan hasil evaluasi model ke database."""
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO model_results (model_name, r2_score, rmse_score, is_best)
    VALUES (?, ?, ?, ?)
    """, (name, r2, rmse, is_best))
    conn.commit()

def save_data_info(conn, key, value):
    """Menyimpan info data ke database."""
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO data_info (key, value)
    VALUES (?, ?)
    """, (key, value))
    conn.commit()


# =====================================================
# FUNGSI UTAMA: PELATIHAN
# =====================================================
def train_and_save_models():
    """Melatih model, mengevaluasi, menyimpan model terbaik, dan mengisi DB."""
    conn = get_db_connection()
    
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: File CSV '{DATA_PATH}' tidak ditemukan.")
        return

    # --- 1. Pra-Pemrosesan Data ---
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Simpan info data ke DB
    save_data_info(conn, 'total_data', len(df))
    save_data_info(conn, 'train_data', len(X_train))
    save_data_info(conn, 'test_data', len(X_test))

    # Transformer: Hanya StandardScaler karena fitur yang diminta numerik
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.columns)
        ],
        remainder='passthrough'
    )

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost Regressor': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }

    best_r2 = -np.inf
    best_pipeline = None
    best_model_name = ""
    
    # Hapus hasil model sebelumnya sebelum menyimpan yang baru
    conn.execute("DELETE FROM model_results")

    print("--- Memulai Pelatihan Model ---")
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', model)])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        is_best = 0
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline
            is_best = 1 # Tandai model terbaik di iterasi ini
        
        save_model_result(conn, name, r2, rmse, is_best)
        print(f"Model: {name} | R2: {r2:.4f} | RMSE: {rmse:.2f}")

    # --- 2. Simpan Model Terbaik ---
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    print(f"\n✅ Model terbaik '{best_model_name}' (R2: {best_r2:.4f}) berhasil disimpan ke '{BEST_MODEL_PATH}'")
    
    # --- 3. Perhitungan dan Penyimpanan SHAP ---
    print("--- Menghitung dan Menyimpan Aset SHAP ---")
    
    best_regressor = best_pipeline['regressor']
    
    # Ambil data sampel training yang sudah diproses untuk SHAP
    X_train_processed = best_pipeline['preprocessor'].transform(X_train)
    X_train_df_processed = pd.DataFrame(X_train_processed, columns=FEATURE_COLUMNS)

    # Ambil 100 baris acak dari data yang diproses untuk SHAP (mengurangi beban)
    X_shap_sample = X_train_df_processed.sample(min(100, len(X_train_df_processed)), random_state=42)

    # Gunakan SHAP.Explainer modern
    explainer = shap.Explainer(best_regressor, X_shap_sample)

    # Simpan Explainer, data sampel, dan nama fitur ke dalam satu package
    shap_package = {
        'explainer': explainer,
        'X_shap_sample': X_shap_sample,
        'feature_names': FEATURE_COLUMNS
    }
    joblib.dump(shap_package, SHAP_PATH)
    print(f"✅ SHAP Explainer package berhasil disimpan ke '{SHAP_PATH}'")

    conn.close()


# ======================================================
# FUNGSI TAMBAHAN (KREASI)
# ======================================================

def predict_score(sleep_hours, studied_hours):
    """Memuat model terbaik dan memprediksi skor."""
    try:
        pipeline = joblib.load(BEST_MODEL_PATH)
    except FileNotFoundError:
        return None
    
    new_data = pd.DataFrame({
        'Sleep_Hours': [sleep_hours],
        'Hours_Studied': [studied_hours]
    })
    
    pred = pipeline.predict(new_data)[0]
    return pred


def generate_recommendation(score):
    """Memberikan rekomendasi berdasarkan skor prediksi."""
    if score >= 86:
        return "🟢 A — Sangat Baik. Pertahankan ritme belajar & tidur yang sudah efektif."
    elif score >= 71:
        return "🔵 B — Baik. Targetkan penambahan 30–45 menit belajar per hari untuk hasil optimal."
    elif score >= 56:
        return "🟡 C — Cukup. Perlu fokus pada peningkatan kualitas tidur (min 7 jam) dan latihan soal."
    else:
        return "🔴 D — Kurang. Perlu tindakan korektif cepat: evaluasi pola tidur dan mulai bimbingan intensif."

def find_optimal_hours(current_sleep, current_study, target_score=86):
    """Mencari kombinasi Jam Tidur dan Jam Belajar optimal di sekitar input."""
    
    if predict_score(current_sleep, current_study) >= target_score:
        return None, None, None # Sudah optimal

    best_score = 0
    best_sleep = 0
    best_study = 0

    # Cari di kisaran +/- 1 jam tidur dan +/- 2 jam belajar
    sleep_range = np.linspace(max(5, current_sleep - 1), min(10, current_sleep + 1), 10)
    study_range = np.linspace(max(1, current_study - 2), min(15, current_study + 2), 10)

    for s in sleep_range:
        for t in study_range:
            score = predict_score(s, t)
            if score is not None and score > best_score:
                best_score = score
                best_sleep = s
                best_study = t

    if best_score > predict_score(current_sleep, current_study):
        return best_sleep, best_study, best_score
    return None, None, None


if __name__ == '__main__':
    train_and_save_models()