# pages/3_🤖_Real_Time_Predictions.py 

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from styles import inject_global_styles, page_header
inject_global_styles()
page_header("🤖 Prédictions Temps Réel", "Étape 3/6 - BRUT + NORMALISÉ + ALERTES 5G")

from pathlib import Path
import sys
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'utils'))
sys.path.insert(0, str(BASE_DIR))

from utils.predictor import TrafficPredictor

# =========================
# VÉRIFICATIONS
# =========================
required_keys = [
    "model_results", "trained_models", "X_test", "y_test",
    "target_scaler", "targets", "features", "prediction_horizon",
    "sequence_length"
]
missing = [k for k in required_keys if k not in st.session_state or st.session_state[k] is None]

if missing:
    st.error(f"❌ **Page 2 incomplète** : {', '.join(missing)} manquantes")
    st.stop()

# =========================
# RÉCUPÉRATION
# =========================
model_results = st.session_state.model_results
trained_models = st.session_state.trained_models
X_test = st.session_state.X_test
y_test = st.session_state.y_test
target_scaler = st.session_state.target_scaler
targets = st.session_state.targets
features = st.session_state.features
base_prediction_horizon = st.session_state.prediction_horizon
sequence_length = st.session_state.sequence_length
n_targets = len(targets)

st.success(f"✅ **{n_targets} cibles** : {targets}")

# =========================
# CONFIGURATION
# =========================
st.subheader("🎯 **Configuration**")
col1, col2 = st.columns(2)
selected_model = col1.selectbox("🤖 Modèle", model_results['Model'].tolist())
horizon = col2.slider(
    "🔮 Horizon (s)",
    1, int(base_prediction_horizon),
    int(base_prediction_horizon)
)

# =========================
# FONCTION PRÉDICTION
# =========================
def run_predictions(horizon: int):
    """
    Utilise TrafficPredictor pour générer:
    - prédiction multi-modèles (normalisées)
    - prédiction du modèle sélectionné
    """
    # Dernière séquence de test
    X_last = X_test[-1:]

    predictor = TrafficPredictor(
        models_dict=trained_models,
        features=features,
        targets=targets,
        sequence_length=sequence_length,
        prediction_horizon=base_prediction_horizon
    )
    all_preds = predictor.predict_all(X_last, horizon=horizon)

    # Filtrer sur le modèle choisi
    pred_scaled = all_preds.get(selected_model)
    if pred_scaled is None:
        return None, None

    # Tronquer au cas où horizon < base_prediction_horizon
    pred_scaled = pred_scaled[:, :horizon, :]

    # y_test est déjà normalisé → on coupe aussi
    y_true_scaled = y_test[-1:, :horizon, :]

    return y_true_scaled, pred_scaled

# =========================
# PRÉDICTIONS ✅ MULTI-CIBLES
# =========================
if st.button("🔮 **LANCER PRÉDICTIONS**", type="primary", use_container_width=True):
    with st.spinner("Calcul en cours..."):
        y_true_scaled, pred_scaled = run_predictions(horizon)

        if pred_scaled is None or y_true_scaled is None:
            st.error("❌ Erreur prédiction (modèle non disponible ou sortie invalide).")
            st.stop()

        # 1. NORMALISÉ → TOUTES LES CIBLES
        st.subheader("🔍 **NORMALISÉ (Debug Modèle)**")
        for i, target_name in enumerate(targets):
            norm_df = pd.DataFrame({
                'Seconde': range(1, horizon + 1),
                'RÉEL_norm': y_true_scaled[0, :horizon, i].round(4),
                f'PRÉDIT_norm_{target_name}': pred_scaled[0, :horizon, i].round(4),
                'Erreur_norm': np.abs(
                    y_true_scaled[0, :horizon, i] - pred_scaled[0, :horizon, i]
                ).round(4)
            })
            st.dataframe(norm_df, use_container_width=True)

        # 2. BRUT → TOUTES LES CIBLES
        st.subheader("🚀 **BRUT - BASE DÉCISION**")

        pred_real_flat = target_scaler.inverse_transform(
            pred_scaled.reshape(-1, n_targets)
        )
        pred_real = pred_real_flat.reshape(pred_scaled.shape)

        y_true_real_flat = target_scaler.inverse_transform(
            y_true_scaled.reshape(-1, n_targets)
        )
        y_true_real = y_true_real_flat.reshape(1, horizon, n_targets)

        for i, target_name in enumerate(targets):
            real_df = pd.DataFrame({
                'Seconde': range(1, horizon + 1),
                f'RÉEL_{target_name}': y_true_real[0, :horizon, i].round(2),
                f'PRÉDIT_{target_name}': pred_real[0, :horizon, i].round(2),
                f'Erreur_{target_name}': np.abs(
                    y_true_real[0, :horizon, i] - pred_real[0, :horizon, i]
                ).round(2)
            })
            st.dataframe(real_df, use_container_width=True)

        # 3. GRAPHIQUES → TOUTES LES CIBLES
        st.subheader("📈 **Graphes BRUTS**")
        cols = st.columns(min(2, len(targets)))
        for i, target_name in enumerate(targets):
            with cols[i % len(cols)]:
                fig = go.Figure()
                x_h = list(range(1, horizon + 1))
                fig.add_trace(
                    go.Scatter(
                        x=x_h,
                        y=y_true_real[0, :horizon, i],
                        name=f"{target_name} RÉEL",
                        line=dict(color="blue", width=3)
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_h,
                        y=pred_real[0, :horizon, i],
                        name=f"{target_name} PRÉDIT",
                        line=dict(color="orange", width=3, dash="dash")
                    )
                )
                fig.update_layout(
                    height=350,
                    title=f"{target_name} (BRUT)"
                )
                st.plotly_chart(fig, use_container_width=True)

        # 4. ALERTES 5G (MULTI-CIBLES)
        st.subheader("🚨 **ALERTES RÉSEAU 5G**")
        metrics = []
        for i, target_name in enumerate(targets):
            pred_mean = float(np.mean(pred_real[0, :horizon, i]))
            mae = float(np.mean(np.abs(
                y_true_real[0, :horizon, i] - pred_real[0, :horizon, i]
            )))
            metrics.append((target_name, pred_mean, mae))

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "📡 Moyenne Prédite",
            f"{np.mean([m[1] for m in metrics]):.1f}"
        )
        col2.metric(
            "📉 MAE Moyen",
            f"{np.mean([m[2] for m in metrics]):.2f}"
        )

        status = "🟢 NOMINAL" if all(m[1] > 5 for m in metrics) else "🟡 DÉGRADATION"
        col3.metric("🚨 Statut", status)

        st.success("✅ **PRÊT POUR DÉCISION 5G**")

# =========================
# RÉSUMÉ TEXTE
# =========================
st.subheader("📋 **Décision Opérateur 5G**")
st.info("""
**✅ BASE DÉCISION = VALEURS BRUTES (toutes cibles)**  
- **>50** → Saturation → Ajouter capacité  
- **<5** → Dégradation → Diagnostic RAN  
- **5-50** → Nominal → Surveillance
""")
