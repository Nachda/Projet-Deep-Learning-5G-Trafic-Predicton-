# utils/predictor.py 
import numpy as np
import pandas as pd
import streamlit as st
from utils.model_trainer import ModelTrainer


class TrafficPredictor:
    """Prédicteur multi-modèles 5G, compatible baselines / ML / DL."""

    def __init__(self, models_dict, features, targets, sequence_length=None, prediction_horizon=None):
        """
        models_dict : dict {nom_modèle: objet modèle ou fonction baseline}
        features : liste des colonnes features
        targets : liste des colonnes cibles
        sequence_length / prediction_horizon : optionnels (pour info)
        """
        self.models = models_dict
        self.features = features
        self.targets = targets
        self.n_targets = len(targets)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

    def _ensure_array(self, X_input):
        """Force X_input en np.ndarray 3D (1, seq_len, n_features)."""
        if isinstance(X_input, pd.DataFrame):
            X_input = X_input.values
        X_input = np.asarray(X_input, dtype=np.float32)

        if X_input.ndim == 2:
            # (seq_len, n_features) -> (1, seq_len, n_features)
            X_input = X_input[np.newaxis, :, :]
        return X_input

    def _predict_single_model(self, model_name, model_obj, X_input, horizon):
        """
        Retourne y_pred de shape (batch, horizon, n_targets)
        ou None si erreur.
        """
        try:
            X_input = self._ensure_array(X_input)
            batch, seq_len, n_features = X_input.shape

            # Baselines (fonctions)
            if model_name == 'Persistence (baseline)':
                # attendu: fonction(X, n_targets, horizon)
                if callable(model_obj):
                    return model_obj(X_input, self.n_targets, horizon)
                return None

            if model_name == 'Moving Average (baseline)':
                if callable(model_obj):
                    return model_obj(X_input, self.n_targets, horizon)
                return None

            # Ensembles stockés comme list/dict -> gérés au niveau de la page 2,
            # ici on ne prédit pas directement avec eux
            if isinstance(model_obj, list) or isinstance(model_obj, dict):
                return None

            # Modèles ML (scikit-learn / XGBoost)
            if hasattr(model_obj, 'predict') and not (
                'keras' in str(type(model_obj)).lower()
            ):
                X_flat = X_input.reshape(X_input.shape[0], -1)
                pred_flat = model_obj.predict(X_flat)
                pred_flat = np.asarray(pred_flat, dtype=np.float32)
                # On suppose que la forme correspond bien à (batch, horizon * n_targets)
                if pred_flat.ndim == 2 and pred_flat.shape[1] == horizon * self.n_targets:
                    return pred_flat.reshape(batch, horizon, self.n_targets)
                # Ou alors déjà (batch, horizon, n_targets)
                if pred_flat.ndim == 3:
                    return pred_flat
                # Sinon, on ne sait pas reshape correctement
                return None

            # Modèles DL (Keras / tf.keras)
            if hasattr(model_obj, 'predict') and 'keras' in str(type(model_obj)).lower():
                # Assurer un shape statique en numpy -> tf gère bien
                X_tf = np.asarray(X_input, dtype=np.float32)
                pred_flat = model_obj.predict(X_tf, verbose=0)
                pred_flat = np.asarray(pred_flat, dtype=np.float32)

                # Sortie (batch, horizon * n_targets)
                if pred_flat.ndim == 2 and pred_flat.shape[1] == horizon * self.n_targets:
                    return pred_flat.reshape(batch, horizon, self.n_targets)
                # Sortie (batch, horizon, n_targets)
                if pred_flat.ndim == 3:
                    return pred_flat
                return None

            # Autres cas non gérés
            return None

        except Exception as e:
            st.warning(f"⚠️ {model_name}: {e}")
            return None

    def predict_all(self, X_input, horizon=10):
        """Prédictions de tous les modèles disponibles."""
        X_input = self._ensure_array(X_input)
        predictions = {}

        for model_name, model in self.models.items():
            pred = self._predict_single_model(model_name, model, X_input, horizon)
            predictions[model_name] = pred

        return predictions

    def predict_ensemble(self, X_input, method='weighted', horizon=10):
        """Ensemble des prédictions (sur les modèles valides)."""
        preds = self.predict_all(X_input, horizon)
        valid_arrays = [v for v in preds.values() if v is not None]

        if not valid_arrays:
            return None

        stacked = np.stack(valid_arrays, axis=0)  # (n_models, batch, horizon, n_targets)

        if method == 'weighted':
            weights = np.ones(stacked.shape[0]) / stacked.shape[0]
            weights = weights[:, np.newaxis, np.newaxis, np.newaxis]
            ensemble = np.sum(stacked * weights, axis=0)
        else:  # median
            ensemble = np.median(stacked, axis=0)

        return ensemble
