# pages/4_📈_Model_Analysis.py 

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from styles import inject_global_styles, page_header
inject_global_styles()
page_header("📈 Analyse & Comparaison", "Étape 4/6 - Benchmark Détaillé")

from pathlib import Path
import sys
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'utils'))
sys.path.insert(0, str(BASE_DIR))

from utils.analyzer import ModelAnalyzer
from utils.visualizer import ComparisonVisualizer

# =========================
# VÉRIFICATIONS
# =========================
required_keys = [
    "models_trained", "model_results", "trained_models",
    "X_test", "y_test", "targets", "target_scaler", "prediction_horizon"
]
missing = [k for k in required_keys if k not in st.session_state or st.session_state[k] is None]

if missing:
    st.error(f"❌ **Pages précédentes incomplètes** : {', '.join(missing)} manquantes")
    st.stop()

results_df = st.session_state.model_results.copy()
trained_models = st.session_state.trained_models
X_test = st.session_state.X_test
y_test = st.session_state.y_test
targets = st.session_state.targets
target_scaler = st.session_state.target_scaler
prediction_horizon = st.session_state.prediction_horizon
n_targets = len(targets)

st.success(f"✅ **{len(results_df)} modèles** | **{n_targets} cibles** : {targets}")

analyzer = ModelAnalyzer(model_results=results_df)

# =========================
# 1. SCORE COMPOSITE
# =========================
st.subheader("🎯 **Score Composite**")

results_df = analyzer.calculate_composite_score(
    df_results=results_df,
    weights={'MAE': 0.4, 'R2': 0.4, 'Train_Time_s': 0.2}
)
composite_df = results_df.sort_values('Composite_Score', ascending=False)[
    ['Model', 'MAE', 'R2', 'Train_Time_s', 'Composite_Score']
]
st.dataframe(composite_df.head(10).round(4), use_container_width=True)

top3 = composite_df.head(3)['Model'].tolist()

# =========================
# 2. VISUALISATIONS GLOBALes
# =========================
col1, col2 = st.columns(2)
with col1:
    radar_fig = ComparisonVisualizer.plot_radar_chart(
        df_metrics=composite_df.head(5),
        metrics=['MAE', 'R2', 'Train_Time_s'],
        title="Radar Score Composite (Top 5)"
    )
    if radar_fig is not None:
        st.plotly_chart(radar_fig, use_container_width=True)

with col2:
    fig_scatter = px.scatter(
        results_df,
        x='MAE', y='R2',
        size='Train_Time_s',
        color='Composite_Score',
        hover_name='Model',
        color_continuous_scale='RdYlGn_r'
    )
    fig_scatter.update_layout(title="MAE vs R² (taille = Train_Time_s)")
    st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("🧩 **Matrice de Performance**")
heatmap_fig = ComparisonVisualizer.plot_performance_matrix(
    df_results=results_df[['Model', 'MAE', 'R2', 'Train_Time_s']],
    title="Matrice normalisée MAE / R² / Temps"
)
if heatmap_fig is not None:
    st.plotly_chart(heatmap_fig, use_container_width=True)

# =========================
# 3. TOP 3 : RÉEL vs PRÉDIT 
# =========================
st.subheader("🎬 **Top 3 : Réel vs Prédit sur le dernier horizon**")

# Dernière séquence test
X_last = X_test[-1:].astype(np.float32)
y_last = y_test[-1:]  # shape (1, horizon, n_targets)

# Dénormalisation du dernier horizon complet
y_last_real = target_scaler.inverse_transform(
    y_last.reshape(-1, n_targets)
).reshape(1, prediction_horizon, n_targets)

fig = make_subplots(
    rows=n_targets, cols=1,
    subplot_titles=[f"{target} (dernier horizon)" for target in targets],
    vertical_spacing=0.1
)

for target_idx, target_name in enumerate(targets):
    # Réel sur horizon
    y_real_h = y_last_real[0, :prediction_horizon, target_idx]

    fig.add_trace(
        go.Scatter(
            x=list(range(1, prediction_horizon + 1)),
            y=y_real_h,
            name='Réel',
            line=dict(color='blue'),
            showlegend=(target_idx == 0)
        ),
        row=target_idx + 1, col=1
    )

    # Prédictions des top 3 sur la même fenêtre
    for i, model_name in enumerate(top3[:3]):
        model = trained_models.get(model_name)
        if model is None:
            continue

        try:
            # Keras vs ML
            if 'keras' in str(type(model)).lower():
                y_pred_flat = model.predict(X_last, verbose=0)
            else:
                X_flat = X_last.reshape(1, -1)
                y_pred_flat = model.predict(X_flat)

            # y_pred_scaled : (1, horizon, n_targets)
            y_pred_scaled = y_pred_flat.reshape(1, prediction_horizon, n_targets)
            y_pred_real = target_scaler.inverse_transform(
                y_pred_scaled.reshape(-1, n_targets)
            ).reshape(1, prediction_horizon, n_targets)[0, :, target_idx]

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, prediction_horizon + 1)),
                    y=y_pred_real,
                    name=model_name.split()[0],
                    line=dict(
                        color=f'rgb({100+80*i},{200-40*i},150)',
                        dash='dash'
                    ),
                    showlegend=(target_idx == 0)
                ),
                row=target_idx + 1, col=1
            )
        except Exception:
            continue

fig.update_layout(
    height=300 * n_targets,
    title="Top 3 Réel vs Prédit (dernier horizon)",
    showlegend=True
)
st.plotly_chart(fig, use_container_width=True)

# =========================
# 4. EXPORT
# =========================
col1, col2 = st.columns(2)
with col1:
    csv = composite_df.round(4).to_csv(index=False)
    st.download_button("📥 CSV (Scores)", csv, "models_scores.csv", "text/csv")

with col2:
    if st.button("⚡ **Page 5 : Actions 5G** →", type="primary"):
        st.switch_page("pages/5_⚡_Network_Actions.py")
