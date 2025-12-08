"""
dashboard_app.py
VERSI FINAL ROBUST:
- Menambahkan pengecekan keberadaan kolom dan data kosong sebelum st.dataframe.
- Menggunakan sintaks SHAP modern yang benar.
- Struktur 7 Tab Lengkap.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
import shap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime

# Impor utilities dari file backend (Pastikan performance_predictor.py sudah ada)
from performance_predictor import get_db_connection, predict_score, generate_recommendation 

# =====================================================
# KONFIGURASI
# =====================================================
DB_NAME = "student_performance.db"
MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
SHAP_PATH = os.path.join(MODEL_DIR, "shap_explainer_package.joblib")
DATA_PATH = "StudentPerformanceFactors.csv"
FEATURE_COLUMNS = ["Sleep_Hours", "Hours_Studied"]

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# =====================================================
# UTILITIES & MEMUAT ASET
# =====================================================

@st.cache_resource
def load_assets():
    """Memuat model terbaik dan package SHAP."""
    try:
        pipeline = joblib.load(BEST_MODEL_PATH)
        
        if os.path.exists(SHAP_PATH):
            shap_package = joblib.load(SHAP_PATH)
            return pipeline, shap_package
        else:
            return pipeline, None
            
    except FileNotFoundError:
        st.error("File model/data tidak ditemukan. Pastikan Anda sudah menjalankan 'performance_predictor.py' terlebih dahulu.")
        st.stop()
    except Exception as e:
        st.error(f"Gagal memuat model/aset: {e}")
        st.stop()


def load_data(conn, query):
    """Memuat data dari SQLite."""
    try:
        return pd.read_sql(query, conn)
    except pd.io.sql.DatabaseError as e:
        # Jika tabel prediction_history belum dibuat/kosong, kembalikan DataFrame kosong
        return pd.DataFrame()


# =====================================================
# APLIKASI UTAMA
# =====================================================

def app():
    if 'guidance_threshold' not in st.session_state:
        # Nilai default awal, misalnya 75
        st.session_state.guidance_threshold = 75
    pipeline, shap_package = load_assets()
    conn = get_db_connection()
    
    st.title("Dashboard — Student Performance Prediction")
    # st.caption("Advanced Modeling • Optimization • SHAP Explainability • Database Log")

    # Muat Data Info dan Model Results
    data_info_df = load_data(conn, "SELECT * FROM data_info")
    results_df = load_data(conn, "SELECT * FROM model_results").set_index('model_name')
    # MUAT HISTORY DENGAN ROBUSTNESS
    history_df = load_data(conn, "SELECT * FROM prediction_history")
    if not history_df.empty:
        history_df = history_df.sort_values(by='id', ascending=False)
    
    # Ambil Metrik Utama
    total_data = data_info_df.loc[data_info_df['key'] == 'total_data', 'value'].iloc[0]
    best_model_row = results_df[results_df['is_best'] == 1].iloc[0]

    # --- HEADER METRICS ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Data", value=f"{int(total_data)} Sampel")
    with col2:
        st.metric(label="Data Training", value=f"{int(data_info_df.loc[data_info_df['key'] == 'train_data', 'value'].iloc[0])} Sampel")
    with col3:
        st.metric(label="Model Terbaik", value=best_model_row.name)
    with col4:
        st.metric(label="R² Score Terbaik", value=f"{best_model_row['r2_score']:.4f}", 
                  delta=f"RMSE: {best_model_row['rmse_score']:.2f}")

    st.markdown("---")

    # =====================================================
    # TABS
    # =====================================================
    tabs = st.tabs([
        "🏠 Overview",
        "🤖 Prediksi Manual",
        "📊 Hasil Pelatihan",
        "💡 Interpretasi (SHAP)",
        "🎯 Optimasi & Rekomendasi",
        "📋 History & Bimbingan"
    ])

    # =====================================================
    # TAB 1 — OVERVIEW
    # =====================================================
    with tabs[0]:
        st.header("Overview Dataset & Kinerja Model")
        
        # Grafik 3D
        st.subheader("Hubungan Jam Tidur, Jam Belajar, dan Nilai Ujian")
        try:
            df_full = pd.read_csv(DATA_PATH)
            fig = px.scatter_3d(df_full, x='Hours_Studied', y='Sleep_Hours', z='Exam_Score',
                              color='Exam_Score', color_continuous_scale=px.colors.sequential.Viridis,
                              title='Data Asli: Hubungan 3D')
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("File CSV 'StudentPerformanceFactors.csv' tidak ditemukan untuk visualisasi.")

        st.subheader("Perbandingan Kinerja Model")
        st.dataframe(results_df.rename(columns={'r2_score':'R² Score', 'rmse_score':'RMSE'}), use_container_width=True)

    # =====================================================
    # TAB 2 — PREDIKSI MANUAL
    # =====================================================
    with tabs[1]:
        st.header("Prediksi Skor Ujian Akhir")
        
        col_input, col_pred = st.columns([1, 2])
        
        with col_input:
            input_hours = st.slider("🕰️ Jam Belajar (Hours_Studied)", min_value=1, max_value=15, value=7, step=1)
            input_sleep = st.slider("😴 Jam Tidur (Sleep_Hours)", min_value=5, max_value=10, value=7, step=1)
            
            if st.button("Hitung Prediksi Skor", key="predict_btn", type="primary"):
                predicted_score = predict_score(input_sleep, input_hours)
                
                if predicted_score is not None:
                    st.success(f"## ✅ Hasil Prediksi Skor: **{predicted_score:.2f}**")
                    st.markdown(f"**Rekomendasi:** {generate_recommendation(predicted_score)}")
                    
                    # Simpan ke History DB
                    cursor = conn.cursor()
                    cursor.execute("""
                    INSERT INTO prediction_history (sleep_hours, studied_hours, predicted_score, timestamp)
                    VALUES (?, ?, ?, ?)
                    """, (input_sleep, input_hours, predicted_score, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                    conn.commit()
                    
                else:
                    st.error("Gagal memuat model untuk prediksi.")

        with col_pred:
            st.subheader("Visualisasi Prediksi Terbaru")
            
            if 'predicted_score' in locals():
                try:
                    df_full = pd.read_csv(DATA_PATH)
                    fig_scatter = go.Figure(data=[
                        go.Scatter3d(
                            x=df_full['Hours_Studied'], y=df_full['Sleep_Hours'], z=df_full['Exam_Score'],
                            mode='markers', marker=dict(size=3, color='gray'), name='Data Asli'
                        ),
                        go.Scatter3d(
                            x=[input_hours], y=[input_sleep], z=[predicted_score],
                            mode='markers', marker=dict(size=8, color='red'), name='Prediksi Anda'
                        )
                    ])
                    fig_scatter.update_layout(scene=dict(
                        xaxis_title='Jam Belajar', yaxis_title='Jam Tidur', zaxis_title='Skor Ujian'
                    ), title="Prediksi Anda di Konteks Data")
                    st.plotly_chart(fig_scatter, use_container_width=True)
                except:
                    st.info("Masukkan input dan klik 'Hitung Prediksi Skor' untuk visualisasi (membutuhkan file CSV).")
            else:
                st.info("Masukkan input dan klik 'Hitung Prediksi Skor' untuk visualisasi.")

    # =====================================================
    # TAB 3 — HASIL PELATIHAN
    # =====================================================
    with tabs[2]:
        st.header("Ringkasan Metrik dan Perbandingan Model")
        
        st.subheader("Metrik Evaluasi Model")
        results_display = results_df.rename(columns={
            'r2_score': 'R² Score', 'rmse_score': 'RMSE', 'is_best': 'Model Terbaik'
        })
        results_display['Model Terbaik'] = results_display['Model Terbaik'].replace({1: 'Ya', 0: 'Tidak'})
        st.dataframe(results_display.sort_values(by='R² Score', ascending=False), use_container_width=True)
        
        st.subheader("Perbandingan R² Score")
        fig_r2 = px.bar(results_df.reset_index(), x='model_name', y='r2_score', 
                       title='R² Score Antar Model',
                       color='r2_score', 
                       color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(fig_r2, use_container_width=True)

    # =====================================================
    # TAB 4 — SHAP EXPLAINABILITY
    # =====================================================
    with tabs[3]:
        st.header("🧠 Model Explainability — SHAP")
        st.markdown(f"Analisis ini menjelaskan bagaimana **Jam Tidur** dan **Jam Belajar** mempengaruhi prediksi skor oleh model **{best_model_row.name}**.")

        if shap_package is None:
            st.warning("SHAP Explainer belum tersedia. Jalankan 'performance_predictor.py' terlebih dahulu.")
        else:
            try:
                explainer = shap_package["explainer"]
                X_shap_sample = shap_package["X_shap_sample"]
                feature_names = shap_package["feature_names"]
                
                shap_obj = explainer(X_shap_sample)

                # Plot 1: Summary Bar Plot (Feature Importance)
                plt.clf()
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_obj, plot_type="bar", feature_names=feature_names, show=False)
                ax.set_title("Rata-rata Pengaruh Fitur (Feature Importance)")
                st.pyplot(fig)
                
                st.markdown("---")
                
                # Plot 2: Summary Dot Plot (Pengaruh positif/negatif)
                plt.clf()
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_obj, feature_names=feature_names, show=False)
                st.pyplot(fig2)
                st.caption("Plot ini menunjukkan apakah nilai fitur tinggi/rendah (warna) mempengaruhi prediksi secara positif/negatif (sumbu X).")
                
            except Exception as e:
                st.error(f"Gagal memproses SHAP Explainer. Error: {e}")

    # =====================================================
    # TAB 5 — OPTIMASI & REKOMENDASI
    # =====================================================
    with tabs[4]:
        st.header("🎯 Analisis Optimasi Jam Belajar & Tidur")
        
        sleep_range = np.linspace(5, 10, 20)
        study_range = np.linspace(1, 15, 20)
        
        def predict_for_heatmap(s, t):
            score = predict_score(s, t)
            return score if score is not None else 0

        data = [[predict_for_heatmap(s, t) for t in study_range] for s in sleep_range]
        data_np = np.array(data)
        
        max_index = np.unravel_index(np.argmax(data_np, axis=None), data_np.shape)
        best_sleep = sleep_range[max_index[0]]
        best_study = study_range[max_index[1]]
        max_score = data_np[max_index[0]][max_index[1]]

        st.success(f"Rekomendasi Optimal Global: Tidur **{best_sleep:.1f} jam**, Belajar **{best_study:.1f} jam** → Nilai **{max_score:.2f}**")

        fig = px.imshow(data,
                        x=[f"{t:.1f}" for t in study_range],
                        y=[f"{s:.1f}" for s in sleep_range],
                        labels=dict(x="Jam Belajar", y="Jam Tidur", color="Nilai"),
                        title="Heatmap Nilai Ujian Berdasarkan Kombinasi Jam Tidur/Belajar")
        st.plotly_chart(fig, use_container_width=True)


    # =====================================================
    # TAB 6 — HISTORY & BIMBINGAN (PERBAIKAN ERROR)
    # =====================================================
    with tabs[5]:
        st.header("📋 Log History dan Analisis Bimbingan")
        
        # Pengaturan Ambang Batas
        st.subheader("Pengaturan Ambang Batas")
       
        # 1. Slider mengambil nilai dari Session State saat ini
        new_threshold = st.slider(
            "Ambang Batas Nilai untuk Bimbingan (Simulasi)", 
            min_value=50, 
            max_value=85, 
            # Menggunakan Session State untuk nilai default
            value=st.session_state.guidance_threshold, 
            step=1,
            key="slider_threshold" # Gunakan key unik
        )

    # 2. Tombol untuk menyimpan perubahan ke Session State
        if st.button("💾 Simpan Ambang Batas", key="save_threshold", type="primary"):
            # Hanya simpan jika nilai berubah
            if new_threshold != st.session_state.guidance_threshold:
                st.session_state.guidance_threshold = new_threshold
                st.success(f"✅ Ambang Batas Bimbingan berhasil disimpan: **{new_threshold}**")
            else:
                st.info("Nilai ambang batas tidak berubah.")
        
        # Ambil nilai threshold yang aktif (sudah disimpan di Session State)
        threshold = st.session_state.guidance_threshold
        st.info(f"Ambang Batas Bimbingan Aktif: **{threshold}**")

        st.markdown("---")
        
        
        # --- ROBUSTNESS CHECK TAMBAHAN PADA HISTORY DATA ---
        expected_cols = ['timestamp', 'studied_hours', 'sleep_hours', 'predicted_score']
        
        if history_df.empty or not all(col in history_df.columns for col in expected_cols):
            st.warning("⚠️ Data History Prediksi tidak ditemukan atau belum lengkap. Silakan coba jalankan Prediksi Manual terlebih dahulu.")
            
        else:
            # Analisis Bimbingan
            st.subheader(f"Kasus Prediksi Butuh Bimbingan (Skor < {threshold})")
            needs_guidance = history_df[history_df['predicted_score'] < threshold]
            
            if not needs_guidance.empty:
                st.metric(label="Total Kasus Butuh Bimbingan", value=f"{len(needs_guidance)} Kasus")
                
                # --- BLOK KODE YANG MENYEBABKAN ERROR PADA FILE ASLI ---
                try:
                    st.dataframe(needs_guidance[expected_cols].rename(columns={
                        'timestamp': 'Waktu', 'studied_hours': 'Belajar (Jam)', 'sleep_hours': 'Tidur (Jam)', 'predicted_score': 'Skor Prediksi'
                    }), use_container_width=True,
                            # caption="Riwayat Prediksi yang hasilnya jatuh di bawah ambang batas bimbingan."
                            )
                except Exception as e:
                    st.error(f"Gagal menampilkan DataFrame 'needs_guidance'. Coba Hapus file '{DB_NAME}' dan jalankan ulang 'performance_predictor.py'.")
                    st.exception(e)

            else:
                st.info("Tidak ada riwayat prediksi yang membutuhkan bimbingan saat ini.")
                
            st.markdown("---")
            
            # Tampilkan History Lengkap
            st.subheader("Riwayat Prediksi Lengkap")
            st.dataframe(history_df[expected_cols].rename(columns={
                        'timestamp': 'Waktu', 'studied_hours': 'Belajar (Jam)', 'sleep_hours': 'Tidur (Jam)', 'predicted_score': 'Skor Prediksi'
                    }), use_container_width=True)


if __name__ == '__main__':
    app()