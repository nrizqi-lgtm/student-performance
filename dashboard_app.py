# dashboard_app.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import shap
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
import os

warnings.filterwarnings("ignore")

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Academic Performance Dashboard", layout="wide")
DB_NAME = "student_performance.db"
MODEL_PATH = "best_pipeline.joblib"
SHAP_EXPL_PATH = "shap_explainer.joblib"
FEATURE_COLUMNS = ["Sleep_Hours", "Hours_Studied"]

# -------------------------
# SESSION STATE DEFAULTS
# -------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # default dark per request
if "threshold" not in st.session_state:
    st.session_state.threshold = 70  # default mapping threshold
if "assets_loaded" not in st.session_state:
    st.session_state.assets_loaded = False

# -------------------------
# THEME INJECTION (dark or light)
# -------------------------
def inject_theme(dark: bool = True):
    if dark:
        root_vars = {
            "--bg": "#0b0c0d",
            "--card": "#111214",
            "--text": "#e9eef6",
            "--subtext": "#bfc8d6",
            "--border": "#202224",
            "--primary": "#4d8cff",
        }
    else:
        root_vars = {
            "--bg": "#f7f7f8",
            "--card": "#ffffff",
            "--text": "#111827",
            "--subtext": "#6b7280",
            "--border": "#e6e7ea",
            "--primary": "#2d7ff9",
        }

    css = f"""
    <style>
    :root {{
      --bg: {root_vars['--bg']};
      --card: {root_vars['--card']};
      --text: {root_vars['--text']};
      --subtext: {root_vars['--subtext']};
      --border: {root_vars['--border']};
      --primary: {root_vars['--primary']};
    }}
    html, body, [data-testid="stAppViewContainer"] {{
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }}

    /* Cards & basic */
    .metric-card, .prediction-card, .modal-card {{
        background: var(--card);
        color: var(--text);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid var(--border);
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }}
    .metric-title {{
        color: var(--subtext);
        font-size: 13px;
    }}
    .metric-value {{
        font-size: 22px;
        font-weight: 700;
        color: var(--text);
    }}

    .stTabs [role="tab"] {{
        border-radius: 8px;
        padding: 10px 16px;
        background: rgba(255,255,255,0.03);
    }}
    .stTabs [role="tab"][aria-selected="true"] {{
        background: var(--primary) !important;
        color: white !important;
    }}

    /* settings button */
    .settings-btn {{
        position: fixed;
        top: 16px;
        right: 18px;
        z-index: 9999;
        padding: 8px 12px;
        background: var(--card);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 10px;
        cursor: pointer;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    }}

    /* modal look */
    dialog::backdrop {{
        background: rgba(0,0,0,0.45);
        backdrop-filter: blur(4px);
    }}
    .modal {{
        border: none;
        width: 380px;
        border-radius: 12px;
        padding: 18px;
        background: var(--card);
        color: var(--text);
        box-shadow: 0 12px 36px rgba(0,0,0,0.3);
    }}
    .modal h3 {{
        margin-top: 0;
    }}
    .modal .apply-btn {{
        background: var(--primary);
        color: white;
        padding: 8px 12px;
        border-radius: 10px;
        border: none;
        width: 100%;
        cursor: pointer;
        font-weight: 600;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# inject initial theme
inject_theme(dark=(st.session_state.theme == "dark"))

# -------------------------
# MODAL HTML (settings)
# -------------------------
st.markdown(
    """
<button class="settings-btn" onclick="document.getElementById('settingsModal').showModal()">⚙️ Pengaturan</button>

<dialog id="settingsModal" class="modal" aria-modal="true">
  <h3>Pengaturan Dashboard</h3>
  <div id="settingsContent"></div>
  <div style="margin-top:10px;">
    <button class="apply-btn" onclick="document.getElementById('settingsModal').close()">Tutup</button>
  </div>
</dialog>
""",
    unsafe_allow_html=True,
)

# -------------------------
# DB & ASSET HELPERS
# -------------------------
@st.cache_resource
def get_db_conn():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    # Ensure tables exist (idempotent)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model_results (
            id INTEGER PRIMARY KEY,
            model_name TEXT UNIQUE,
            r2_score REAL,
            rmse_score REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY,
            sleep_hours REAL,
            studied_hours REAL,
            predicted_score REAL,
            timestamp TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS data_info (
            key TEXT PRIMARY KEY,
            value REAL
        )
        """
    )
    conn.commit()
    return conn

@st.cache_resource
def load_assets_safe():
    """Load model and SHAP explainer; return (pipeline, explainer) or (None, None) and show errors in UI."""
    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error memuat model dari '{MODEL_PATH}': {e}")
        pipeline = None
    try:
        explainer = joblib.load(SHAP_EXPL_PATH)
    except Exception as e:
        st.warning(f"SHAP explainer tidak ditemukan atau gagal dimuat: {e}")
        explainer = None
    return pipeline, explainer

# -------------------------
# Utility: Kurikulum mapping
# -------------------------
def nilai_kurikulum(score: float) -> str:
    """Mapping sesuai Kurikulum Merdeka (Kemendikbud)."""
    try:
        s = float(score)
    except:
        return "Nilai tidak valid"
    if s >= 90:
        return "A — Sangat Baik"
    elif s >= 80:
        return "B — Baik"
    elif s >= 70:
        return "C — Cukup"
    else:
        return "D — Perlu Bimbingan"

# -------------------------
# RENDER SETTINGS FORM INTO MODAL
# (We inject Streamlit components into an empty container that will appear under modal's settingsContent)
# -------------------------
settings_container = st.empty()
with settings_container.container():
    # Place content inside an element that shares the same visual modal area (user clicks button -> modal opens)
    st.markdown("<div id='settingsContent' style='padding-bottom:8px;'>", unsafe_allow_html=True)

    with st.form("settings_form"):
        st.markdown("**Pengaturan Ambang Bimbingan & Tema**")
        th = st.number_input(
            "Ambang Bimbingan (nilai di bawah ini akan dianggap perlu bimbingan)", 
            min_value=0, max_value=100, value=int(st.session_state.threshold), step=1, format="%d"
        )
        theme_choice = st.selectbox("Tema Tampilan", options=["dark", "light"], index=0 if st.session_state.theme == "dark" else 1)
        save_btn = st.form_submit_button("Simpan Pengaturan")
        if save_btn:
            st.session_state.threshold = int(th)
            st.session_state.theme = theme_choice
            # inject new theme and rerun to apply
            inject_theme(dark=(st.session_state.theme == "dark"))
            st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# MAIN APP
# -------------------------
def main():
    conn = get_db_conn()
    pipeline, explainer = load_assets_safe()

    st.title("📊 Academic Performance Dashboard")
    st.markdown("Prediksi **Exam Score** berdasarkan jam belajar & jam tidur — mapping nilai mengikuti *Kurikulum Merdeka*.")

    # Load DB tables
    try:
        results_df = pd.read_sql_query("SELECT * FROM model_results", conn)
    except Exception:
        results_df = pd.DataFrame(columns=["model_name","r2_score","rmse_score"])
    try:
        history_df = pd.read_sql_query("SELECT * FROM prediction_history ORDER BY id DESC", conn)
    except Exception:
        history_df = pd.DataFrame(columns=["id","sleep_hours","studied_hours","predicted_score","timestamp"])
    try:
        info_df = pd.read_sql_query("SELECT * FROM data_info", conn)
    except Exception:
        info_df = pd.DataFrame(columns=["key","value"])

    # Ensure data_info keys exist with defaults if missing
    def ensure_info(k, default):
        if k not in info_df.key.values:
            conn.execute("INSERT OR REPLACE INTO data_info (key, value) VALUES (?, ?)", (k, default))
            conn.commit()
    ensure_info("total_data", 0)
    ensure_info("train_data", 0)
    ensure_info("test_data", 0)
    # reload
    info_df = pd.read_sql_query("SELECT * FROM data_info", conn)

    total_data = int(info_df.loc[info_df.key == "total_data", "value"].iloc[0])
    train_data = int(info_df.loc[info_df.key == "train_data", "value"].iloc[0])
    test_data = int(info_df.loc[info_df.key == "test_data", "value"].iloc[0])

    best_model_name = results_df.sort_values("r2_score", ascending=False).iloc[0]["model_name"] if (not results_df.empty) else "—"
    best_r2 = results_df.sort_values("r2_score", ascending=False).iloc[0]["r2_score"] if (not results_df.empty) else 0.0

    # Metrics cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-title">Total Dataset</div>
          <div class="metric-value">{total_data} siswa</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-title">Data Train</div>
          <div class="metric-value">{train_data} sampel</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-title">Model Terbaik</div>
          <div class="metric-value">{best_model_name}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-title">R² Terbaik</div>
          <div class="metric-value">{best_r2:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["✨ Prediksi", "📈 Perbandingan Model", "🔍 Interpretasi SHAP"])

    # TAB 1: Prediction
    with tab1:
        st.header("✨ Prediksi Skor Ujian")
        left, right = st.columns([1.2, 2])

        with left:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("**Form Input (tanpa slider)**")
            study_hours = st.number_input("Jam Belajar (1–15 jam)", min_value=1, max_value=15, value=7, step=1)
            sleep_hours = st.number_input("Jam Tidur (4–12 jam)", min_value=4, max_value=12, value=7, step=1)
            # optional: allow teacher to input student id or name (not required)
            student_name = st.text_input("Nama Siswa (opsional)", value="")
            pred_btn = st.button("🔮 Prediksi Skor")
            st.markdown("</div>", unsafe_allow_html=True)

            if pred_btn:
                if pipeline is None:
                    st.error("Model belum tersedia. Pastikan file model tersimpan di path yang benar.")
                else:
                    input_df = pd.DataFrame({
                        "Sleep_Hours":[sleep_hours],
                        "Hours_Studied":[study_hours]
                    })
                    try:
                        pred_val = pipeline.predict(input_df)[0]
                    except Exception as e:
                        st.error(f"Gagal melakukan prediksi: {e}")
                        pred_val = None

                    if pred_val is not None:
                        kategori = nilai_kurikulum(pred_val)
                        st.success(f"**Prediksi Skor: {pred_val:.2f}** — {kategori}")

                        # save to DB
                        conn.execute(
                            "INSERT INTO prediction_history (sleep_hours, studied_hours, predicted_score, timestamp) VALUES (?, ?, ?, ?)",
                            (sleep_hours, study_hours, float(pred_val), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        )
                        conn.commit()
                        # reload history
                        history_df = pd.read_sql_query("SELECT * FROM prediction_history ORDER BY id DESC", conn)

        with right:
            st.subheader("📚 Riwayat Prediksi")
            if history_df.empty:
                st.info("Belum ada riwayat prediksi.")
            else:
                df_disp = history_df.rename(columns={
                    "timestamp":"Waktu", "studied_hours":"Belajar (jam)", "sleep_hours":"Tidur (jam)", "predicted_score":"Prediksi"
                })
                st.dataframe(df_disp[["Waktu","Belajar (jam)","Tidur (jam)","Prediksi"]], use_container_width=True, hide_index=True)
                csv = df_disp.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Riwayat (CSV)", data=csv, file_name="prediction_history.csv", mime="text/csv")

        st.markdown("---")
        st.subheader(f"Siswa Berpotensi Butuh Bimbingan (Skor < {st.session_state.threshold})")
        warn_df = history_df[history_df["predicted_score"] < st.session_state.threshold] if not history_df.empty else pd.DataFrame()
        if warn_df.empty:
            st.info("Tidak ada siswa di bawah ambang batas.")
        else:
            st.metric("Jumlah Kasus", len(warn_df))
            st.dataframe(warn_df.rename(columns={"timestamp":"Waktu","predicted_score":"Prediksi"}), use_container_width=True, hide_index=True)

    # TAB 2: Model comparison
    with tab2:
        st.header("📈 Perbandingan & Evaluasi Model")
        if results_df.empty:
            st.info("Belum ada hasil evaluasi model. Jalankan script training untuk mengisi tabel model_results.")
        else:
            df_view = results_df.rename(columns={"model_name":"Model","r2_score":"R² Score","rmse_score":"RMSE"}).set_index("Model")
            st.dataframe(df_view, use_container_width=True)
            fig = px.bar(results_df, x="model_name", y="r2_score", title="Perbandingan R² antar Model", color="r2_score", color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

    # TAB 3: SHAP Interpretation
    with tab3:
        st.header("🔍 Interpretasi Model (SHAP)")
        if pipeline is None or explainer is None:
            st.info("SHAP explainer atau model belum tersedia. Pastikan file SHAP dan model ada.")
        else:
            # create small dummy sample from training-like distribution for visualization
            try:
                dummy = pd.DataFrame(np.random.rand(50, 2), columns=FEATURE_COLUMNS)
                # Attempt to preprocess via pipeline
                if "preprocessor" in pipeline.named_steps:
                    dummy_prep = pipeline["preprocessor"].transform(dummy)
                    dummy_df = pd.DataFrame(dummy_prep, columns=FEATURE_COLUMNS)
                else:
                    dummy_df = dummy
                shap_vals = explainer(dummy_df)
                fig, ax = plt.subplots(figsize=(9,5))
                shap.summary_plot(shap_vals, dummy_df, feature_names=FEATURE_COLUMNS, show=False)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Gagal membuat plot SHAP: {e}")

    # Footer small note
    st.markdown("---")
    st.markdown("**Catatan:** Mapping nilai mengikuti *Kurikulum Merdeka* (Kemendikbud). Sistem ini hanya prediksi — gunakan untuk membantu prioritas bimbingan, bukan penilaian akhir.")

if __name__ == "__main__":
    main()
