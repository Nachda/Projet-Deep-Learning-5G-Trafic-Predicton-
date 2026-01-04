# pages/2_🧠_Model_Training.py - VERSION RÉFÉRENTE AVEC ModelTrainer

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.multioutput import MultiOutputRegressor
import gc
import warnings

warnings.filterwarnings('ignore')

# IMPORTS DL
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.warning("⚠️ TensorFlow non disponible → `pip install tensorflow`")

# Styles + header
from styles import inject_global_styles, page_header
inject_global_styles()
page_header("🧠 Entraînement Modèles", "Étape 2/6 - Benchmark 14 Modèles 5G")

# PATHS & utils
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'utils'))
sys.path.insert(0, str(BASE_DIR))

from utils.model_trainer import ModelTrainer

# VÉRIFICATIONS
required_keys = ["processed_data", "features", "targets", "raw_metrics"]
missing = [k for k in required_keys if k not in st.session_state or st.session_state[k] is None]

if missing:
    st.error(f"❌ **Page 1 incomplète** : {', '.join(missing)} manquantes")
    st.info("💡 Retournez à la page 1 pour préparer le dataset.")
    st.stop()

df_norm = st.session_state.processed_data
raw_df = st.session_state.raw_metrics
features = st.session_state.features
targets = st.session_state.targets

st.info(f"📊 **Dataset préparé** : {len(df_norm):,} intervalles | **{len(features)} features** → **{len(targets)} cibles**")
st.info(f"🎯 **Cibles sélectionnées** : {targets}")

# =========================
# CONFIGURATION UTILISATEUR
# =========================
st.subheader("⚙️ **Configuration de l'Entraînement**")

col1, col2, col3 = st.columns(3)
sequence_length = col1.slider(
    "Longueur séquence (lookback)",
    30, 120, 60,
    help="Nombre de pas de temps historiques utilisés pour prédire"
)
prediction_horizon = col2.slider(
    "Horizon prédiction (steps ahead)",
    1, 30, 10,
    help="Nombre de pas de temps futurs à prédire"
)
test_size = col3.slider("Taille jeu test (%)", 10, 40, 20) / 100

col1, col2 = st.columns(2)
epochs = col1.number_input(
    "Epochs Deep Learning", 10, 200, 30,
    help="Nombre d'itérations d'entraînement pour les réseaux neuronaux"
)
batch_size = col2.selectbox(
    "Batch Size DL", [16, 32, 64, 128], index=1,
    help="Taille des lots pour l'entraînement DL"
)

# =========================
# SÉLECTION DES MODÈLES
# =========================
st.subheader("🤖 **Sélection des Modèles à Entraîner**")

models_to_train = [
    "Persistence (baseline)",
    "Moving Average (baseline)",
    "Linear Regression",
    "Random Forest",
    "XGBoost",
    "Gradient Boosting",
    "LSTM",
    "GRU",
    "BiLSTM",
    "CNN_LSTM",
    "Transformer",
    "MLP",
    "Ensemble Voting",
    "Ensemble Stacking"
]

selected_models = st.multiselect(
    "Choisissez les modèles à comparer :",
    models_to_train,
    default=["XGBoost", "Random Forest", "LSTM", "Gradient Boosting", "Linear Regression"],
    help="Sélectionnez au moins 2 modèles pour une comparaison significative"
)

if len(selected_models) < 2:
    st.warning("⚠️ Sélectionnez au moins 2 modèles pour comparer leurs performances")
    st.stop()

# =========================
# BOUTON D'ENTRAÎNEMENT
# =========================
if st.button("🚀 **Lancer l'Entraînement des Modèles**", type="primary", use_container_width=True):
    with st.spinner(f"🔄 Entraînement de {len(selected_models)} modèles en cours..."):
        try:
            # Vérification colonnes
            all_cols = features + targets
            missing_cols = [col for col in all_cols if col not in df_norm.columns]
            if missing_cols:
                st.error(f"❌ Colonnes manquantes dans le dataset : {missing_cols}")
                st.stop()

            # =========================
            # Séquences via ModelTrainer
            # =========================
            trainer = ModelTrainer(
                data=df_norm,
                sequence_length=sequence_length,
                prediction_horizon=prediction_horizon,
                test_size=test_size
            )

            st.info("📈 Création des séquences temporelles...")
            X_train, X_test, y_train, y_test = trainer.prepare_sequences(features, targets, max_sequences=10000)

            if X_train.size == 0:
                st.error("❌ Pas assez de données pour créer des séquences")
                st.info(f"💡 Réduisez sequence_length ({sequence_length}) ou prediction_horizon ({prediction_horizon})")
                st.stop()

            st.success(
                f"✅ {X_train.shape[0] + X_test.shape[0]} séquences créées : "
                f"Train={X_train.shape[0]}, Test={X_test.shape[0]} | "
                f"X.shape={X_train.shape}, y.shape={y_train.shape}"
            )

            # =========================
            # SCALER sur les targets
            # =========================
            n_targets = len(targets)
            target_scaler = RobustScaler()

            y_train_flat = y_train.reshape(-1, n_targets)
            y_test_flat = y_test.reshape(-1, n_targets)

            y_train_scaled_flat = target_scaler.fit_transform(y_train_flat)
            y_test_scaled_flat = target_scaler.transform(y_test_flat)

            y_train = y_train_scaled_flat.reshape(y_train.shape)
            y_test = y_test_scaled_flat.reshape(y_test.shape)

            st.session_state.target_scaler = target_scaler
            st.success("✅ Scaling des targets terminé")

            # =========================
            # ENTRAÎNEMENT DES MODÈLES
            # =========================
            results = []
            trained_models = {}
            y_preds_cache = {}  # Pour ensembles

            for model_name in selected_models:
                start_time = time.time()
                st.info(f"🔄 Entraînement : {model_name}...")

                try:
                    # Cas ensembles: on les traitera après
                    if model_name in ["Ensemble Voting", "Ensemble Stacking"]:
                        continue

                    # Utiliser ModelTrainer pour tous les autres
                    model, y_pred, metrics = trainer.train_and_evaluate(
                        model_name=model_name,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0
                    )

                    trained_models[model_name] = model
                    y_preds_cache[model_name] = y_pred

                    results.append({
                        'Model': metrics['Model'],
                        'MAE': round(metrics['MAE'], 4),
                        'R2': round(metrics['R2'], 3),
                        'Train_Time_s': round(metrics['Train_Time_s'], 2),
                        'Status': '✅'
                    })

                    st.success(
                        f"✅ {model_name} : MAE={metrics['MAE']:.4f}, "
                        f"R²={metrics['R2']:.3f}, Time={metrics['Train_Time_s']:.1f}s"
                    )

                except Exception as e:
                    st.warning(f"⚠️ Erreur sur {model_name} : {str(e)[:100]}...")
                    results.append({
                        'Model': model_name,
                        'MAE': np.nan,
                        'R2': np.nan,
                        'Train_Time_s': np.nan,
                        'Status': '❌'
                    })

                gc.collect()

            # =========================
            # ENSEMBLES (Voting / Stacking)
            # =========================
            from sklearn.linear_model import LinearRegression as MetaLinearRegression

            if "Ensemble Voting" in selected_models:
                ml_models = ['Linear Regression', 'Random Forest', 'XGBoost', 'Gradient Boosting']
                available = [m for m in ml_models if m in trained_models]
                if len(available) >= 2:
                    X_test_flat = X_test.reshape(X_test.shape[0], -1)
                    preds = []
                    for m_name in available:
                        model_obj = trained_models[m_name]
                        if hasattr(model_obj, 'predict'):
                            y_pred_flat = model_obj.predict(X_test_flat)
                            preds.append(y_pred_flat)
                    if preds:
                        pred_flat_avg = np.mean(preds, axis=0)
                        y_pred_ens = pred_flat_avg.reshape(-1, prediction_horizon, n_targets)

                        mae = float(np.mean(np.abs(y_test - y_pred_ens)))
                        r2 = float(r2_score(y_test.reshape(-1), y_pred_ens.reshape(-1)))
                        train_time = time.time() - start_time

                        trained_models["Ensemble Voting"] = available
                        results.append({
                            'Model': "Ensemble Voting",
                            'MAE': round(mae, 4),
                            'R2': round(r2, 3),
                            'Train_Time_s': round(train_time, 2),
                            'Status': '✅'
                        })

            if "Ensemble Stacking" in selected_models:
                ml_models = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
                available = [m for m in ml_models if m in trained_models]
                if len(available) >= 2:
                    X_train_flat = X_train.reshape(X_train.shape[0], -1)
                    X_test_flat = X_test.reshape(X_test.shape[0], -1)

                    train_base_preds = []
                    test_base_preds = []

                    for m_name in available:
                        model_obj = trained_models[m_name]
                        if hasattr(model_obj, 'predict'):
                            train_base_preds.append(model_obj.predict(X_train_flat))
                            test_base_preds.append(model_obj.predict(X_test_flat))

                    if train_base_preds and test_base_preds:
                        train_meta_X = np.concatenate(train_base_preds, axis=1)
                        test_meta_X = np.concatenate(test_base_preds, axis=1)

                        y_train_meta = y_train.reshape(y_train.shape[0], -1)
                        y_test_meta = y_test.reshape(y_test.shape[0], -1)

                        meta_model = MetaLinearRegression()
                        meta_model.fit(train_meta_X, y_train_meta)

                        pred_meta = meta_model.predict(test_meta_X)
                        y_pred_stack = pred_meta.reshape(y_test.shape)

                        mae = float(np.mean(np.abs(y_test - y_pred_stack)))
                        r2 = float(r2_score(y_test.reshape(-1), y_pred_stack.reshape(-1)))
                        train_time = time.time() - start_time

                        trained_models["Ensemble Stacking"] = {
                            'meta': meta_model,
                            'base': available
                        }

                        results.append({
                            'Model': "Ensemble Stacking",
                            'MAE': round(mae, 4),
                            'R2': round(r2, 3),
                            'Train_Time_s': round(train_time, 2),
                            'Status': '✅'
                        })

            # =========================
            # SAUVEGARDE DES RÉSULTATS
            # =========================
            results_df = pd.DataFrame(results)
            results_df = results_df.dropna().sort_values('MAE').reset_index(drop=True)
            results_df['Rank'] = results_df.index + 1

            st.session_state.model_results = results_df
            st.session_state.trained_models = trained_models
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.sequence_length = sequence_length
            st.session_state.prediction_horizon = prediction_horizon
            st.session_state.models_trained = True

            # =========================
            # AFFICHAGE
            # =========================
            st.markdown("---")
            st.subheader("🏆 **Classement des Modèles**")

            display_cols = ['Rank', 'Model', 'MAE', 'R2', 'Train_Time_s', 'Status']
            st.dataframe(results_df[display_cols], use_container_width=True, height=400)

            if len(results_df) > 0:
                best_model = results_df.iloc[0]
                col1, col2, col3 = st.columns(3)
                col1.metric("🥇 **Meilleur modèle**", best_model['Model'])
                col2.metric("📉 **Meilleur MAE**", f"{best_model['MAE']:.4f}")
                col3.metric("⭐ **Meilleur R²**", f"{best_model['R2']:.3f}")
                st.success(f"✅ Entraînement terminé ! {len(results_df)} modèles comparés.")

            if st.button("🤖 **Page 3 : Prédictions & Visualisation**", type="primary"):
                st.switch_page("pages/3_🤖_Real_Time_Predictions.py")

        except Exception as e:
            st.error(f"❌ Erreur lors de l'entraînement : {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# AFFICHAGE SI DÉJÀ ENTRAÎNÉ
if st.session_state.get('models_trained', False):
    st.markdown("---")
    st.subheader("📊 **Résultats d'Entraînement Existants**")

    results_df = st.session_state.model_results

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Modèles entraînés", len(results_df))
    col2.metric("🎯 Cibles", len(targets))
    col3.metric("⏱️ Horizon", f"{st.session_state.get('prediction_horizon', 'N/A')}s")

    st.dataframe(
        results_df[['Model', 'MAE', 'R2', 'Train_Time_s']].head(10),
        use_container_width=True
    )

    if st.button("➡️ **Passer aux Prédictions**", type="primary"):
        st.switch_page("pages/3_🤖_Real_Time_Predictions.py")
