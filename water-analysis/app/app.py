import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.preprocess import load_scaler, scale_features, potability_label, FEATURES, load_data, get_features_target

MODEL_PATH   = os.path.join(os.path.dirname(__file__), '..', 'model', 'rf_model.pkl')
PLOTS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'plots')
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'water_potability.csv')

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_sc():
    return load_scaler()

@st.cache_data
def load_dataset():
    return load_data(DATASET_PATH)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Water Potability Predictor", page_icon="💧", layout="wide")

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f4f9fd; }
  .big-title   { font-size:2.6rem; font-weight:800; color:#0077b6; text-align:center; margin-bottom:0 }
  .subtitle    { font-size:1.05rem; color:#555; text-align:center; margin-bottom:1.5rem }
  .result-card { border-radius:16px; padding:1.6rem 2rem; text-align:center; margin:1rem 0 }
  .score-num   { font-size:4rem; font-weight:900; margin:0; line-height:1 }
  .score-lbl   { font-size:1.4rem; font-weight:700; margin:4px 0 0 0 }
  .score-desc  { font-size:0.95rem; color:#555; margin:6px 0 0 0 }
  .param-who   { font-size:0.78rem; color:#888 }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">💧 Water Potability Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Random Forest Classifier · Predict whether water is safe to drink</p>', unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
try:
    model  = load_model()
    scaler = load_sc()
    st.success("✅ Model loaded — Accuracy 67% · ROC-AUC 0.69 (limited by dataset size & complexity)")
except Exception as e:
    st.error(f"❌ Model not found. Run `python model/train.py` first.\n{e}")
    st.stop()

# ── WHO safe ranges for reference ─────────────────────────────────────────────
WHO = {
    'ph':              (6.5, 8.5),
    'Hardness':        (0,   300),
    'Solids':          (0,   500),
    'Chloramines':     (0,   4.0),
    'Sulfate':         (0,   250),
    'Conductivity':    (0,   400),
    'Organic_carbon':  (0,   2.0),
    'Trihalomethanes': (0,   80),
    'Turbidity':       (0,   1.0),
}

UNITS = {
    'ph': '–', 'Hardness': 'mg/L', 'Solids': 'mg/L',
    'Chloramines': 'mg/L', 'Sulfate': 'mg/L', 'Conductivity': 'μS/cm',
    'Organic_carbon': 'mg/L', 'Trihalomethanes': 'μg/L', 'Turbidity': 'NTU'
}

RANGES = {
    'ph':              (0.0, 14.0,  7.0,  0.1),
    'Hardness':        (47.0, 323.0, 196.0, 0.5),
    'Solids':          (321.0, 61228.0, 20928.0, 10.0),
    'Chloramines':     (0.35, 13.13,  7.12, 0.05),
    'Sulfate':         (129.0, 481.0, 333.0, 0.5),
    'Conductivity':    (181.0, 753.0,  426.0, 0.5),
    'Organic_carbon':  (0.0,  28.3,  14.3, 0.1),   # changed
    'Trihalomethanes': (0.74, 124.0,  66.4, 0.5),
    'Turbidity':       (0.0,  6.74,  3.97, 0.01),  # changed
}

LABELS = {
    'ph': 'pH', 'Hardness': 'Hardness', 'Solids': 'Total Dissolved Solids',
    'Chloramines': 'Chloramines', 'Sulfate': 'Sulfate',
    'Conductivity': 'Conductivity', 'Organic_carbon': 'Organic Carbon',
    'Trihalomethanes': 'Trihalomethanes', 'Turbidity': 'Turbidity'
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("🔬 Water Sample Parameters")
st.sidebar.markdown("Enter your water sample values:")

inputs = {}
for feat in FEATURES:
    mn, mx, default, step = RANGES[feat]
    who_lo, who_hi = WHO[feat]
    inputs[feat] = st.sidebar.slider(
        f"{LABELS[feat]} ({UNITS[feat]})",
        float(mn), float(mx), float(default), float(step),
        help=f"WHO guideline: {who_lo}–{who_hi} {UNITS[feat]}"
    )

predict_btn = st.sidebar.button("🔍 Predict Potability", use_container_width=True, type="primary")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Model Insights", "ℹ️ Parameter Guide"])

with tab1:
    if predict_btn:
        input_df = pd.DataFrame([inputs])[FEATURES]
        X_scaled = scale_features(input_df, scaler)
        prob     = float(model.predict_proba(X_scaled)[0][1])
        pred     = model.predict(X_scaled)[0]
        verdict, desc, color = potability_label(prob)

        col1, col2 = st.columns([1, 1.8], gap="large")

        with col1:
            st.markdown(f"""
            <div class="result-card" style="background:{color}18; border:2px solid {color}">
                <p class="score-num" style="color:{color}">{prob*100:.1f}%</p>
                <p class="score-lbl" style="color:{color}">{verdict}</p>
                <p class="score-desc">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

            # Probability gauge
            fig_g, ax_g = plt.subplots(figsize=(4, 0.7))
            fig_g.patch.set_alpha(0)
            ax_g.barh([''], [1],    color='#eee',  height=0.5)
            ax_g.barh([''], [prob], color=color,    height=0.5)
            ax_g.set_xlim(0, 1)
            ax_g.axis('off')
            ax_g.text(prob/2, 0, f"{prob*100:.1f}%", va='center', ha='center',
                      fontsize=11, fontweight='bold', color='white')
            st.pyplot(fig_g, use_container_width=True)
            plt.close()

            st.markdown("**Confidence Bands**")
            for lbl, rng, c in [("Potable","≥ 70%","#2ecc71"),("Likely Potable","50–69%","#27ae60"),
                                  ("Uncertain","35–49%","#f39c12"),("Not Potable","< 35%","#e74c3c")]:
                st.markdown(
                    f'<span style="background:{c};color:white;padding:2px 8px;'
                    f'border-radius:4px;font-size:0.8rem">{lbl}</span>&nbsp; {rng}',
                    unsafe_allow_html=True
                )

        with col2:
            st.subheader("📋 Parameter Analysis")

            rows = []
            for feat in FEATURES:
                val = inputs[feat]
                who_lo, who_hi = WHO[feat]
                in_range = who_lo <= val <= who_hi
                status = "✅ OK" if in_range else "⚠️ Outside WHO"
                rows.append({
                    'Parameter': LABELS[feat],
                    'Your Value': round(val, 3),
                    'Unit': UNITS[feat],
                    'WHO Safe Range': f"{who_lo}–{who_hi}",
                    'Status': status
                })
            summary_df = pd.DataFrame(rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # How many params are out of range
            out_of_range = sum(1 for r in rows if "⚠️" in r['Status'])
            if out_of_range == 0:
                st.success("All parameters are within WHO guidelines.")
            elif out_of_range <= 3:
                st.warning(f"{out_of_range} parameter(s) outside WHO guidelines.")
            else:
                st.error(f"{out_of_range} parameters outside WHO guidelines — treatment strongly advised.")

    else:
        st.info("👈 Set water sample parameters in the sidebar and click **Predict Potability**")
        st.markdown("""
        ### What does this app do?
        This app uses a **Random Forest Classifier** trained on the real **Water Potability Dataset**
        (3,276 samples) to predict whether a water sample is safe to drink.

        **Input:** 9 physicochemical parameters measured in the water sample  
        **Output:** Probability of potability (0–100%) with a safety verdict

        | Result | Probability | Meaning |
        |--------|-------------|---------|
        | Potable | ≥ 70% | Safe to drink |
        | Likely Potable | 50–69% | Probably safe, lab verification advised |
        | Uncertain | 35–49% | Treatment recommended |
        | Not Potable | < 35% | Unsafe for drinking |
        """)

with tab2:
    st.subheader("Model Performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy",  "67%")
    m2.metric("ROC-AUC",   "0.69")
    m3.metric("CV ROC-AUC","0.69 ± 0.02")

    st.caption("Note: The dataset has inherent noise and class imbalance (61% not potable), "
               "which limits model performance — this is typical for real-world water quality data.")

    plots = {
        'Feature Importance':  'feature_importance.png',
        'Confusion Matrix':    'confusion_matrix.png',
        'ROC Curve':           'roc_curve.png',
    }
    cols = st.columns(3)
    for col, (title, fname) in zip(cols, plots.items()):
        path = os.path.join(PLOTS_DIR, fname)
        if os.path.exists(path):
            col.image(path, caption=title, use_container_width=True)
        else:
            col.warning(f"Plot not found: {fname}")

    st.markdown("""
    **Model:** `RandomForestClassifier` · 300 trees · `class_weight='balanced'`  
    **Preprocessing:** Median imputation for missing values · StandardScaler  
    **Train/Test Split:** 80/20 stratified  
    **Dataset:** 3,276 real water samples (Kaggle Water Potability Dataset)
    """)

    # Dataset distribution
    st.subheader("Dataset Distribution")
    try:
        df = load_dataset()
        fig_d, axes = plt.subplots(3, 3, figsize=(12, 8))
        axes = axes.flatten()
        for i, feat in enumerate(FEATURES):
            axes[i].hist(df[df['Potability']==0][feat].dropna(), bins=30,
                         alpha=0.6, label='Not Potable', color='#e74c3c', density=True)
            axes[i].hist(df[df['Potability']==1][feat].dropna(), bins=30,
                         alpha=0.6, label='Potable', color='#2ecc71', density=True)
            axes[i].set_title(LABELS[feat], fontsize=9)
            axes[i].tick_params(labelsize=7)
        axes[0].legend(fontsize=8)
        plt.suptitle('Feature Distributions by Potability', fontsize=12, y=1.01)
        plt.tight_layout()
        st.pyplot(fig_d, use_container_width=True)
        plt.close()
    except Exception:
        st.info("Dataset not found for distribution plot.")

with tab3:
    st.subheader("WHO Drinking Water Guidelines")
    guide = {
        "pH":               ("6.5–8.5", "–",      "Acidity/alkalinity. Extreme values affect taste and pipe corrosion."),
        "Hardness":         ("< 300",   "mg/L",   "Calcium & magnesium. High hardness causes scaling."),
        "Total Dissolved Solids": ("< 500", "mg/L","Dissolved minerals. High TDS affects taste and safety."),
        "Chloramines":      ("< 4",     "mg/L",   "Disinfectant. Excess causes health issues."),
        "Sulfate":          ("< 250",   "mg/L",   "Naturally occurring. High levels cause laxative effects."),
        "Conductivity":     ("< 400",   "μS/cm",  "Measure of ion concentration. Correlates with dissolved solids."),
        "Organic Carbon":   ("< 2",     "mg/L",   "Organic matter. Feeds microbial growth and forms by-products."),
        "Trihalomethanes":  ("< 80",    "μg/L",   "Disinfection by-products. Potential carcinogens at high levels."),
        "Turbidity":        ("< 1",     "NTU",    "Cloudiness from particles. High turbidity may harbor pathogens."),
    }
    guide_df = pd.DataFrame(guide, index=['WHO Limit','Unit','Description']).T
    guide_df.index.name = 'Parameter'
    st.dataframe(guide_df, use_container_width=True)
    st.caption("Source: WHO Guidelines for Drinking-water Quality, 4th edition.")
