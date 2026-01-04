# utils/model_trainer.py 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Classe pour l'entraînement des modèles de prédiction (ML + DL + baselines)"""

    def __init__(self, data, sequence_length=60, prediction_horizon=10, test_size=0.2):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.test_size = test_size
        self.models = {}
        self.histories = {}

    # =====================================================================
    # 1) Préparation des données: fonction canonique de séquences
    # =====================================================================
    def prepare_sequences(self, features, targets, max_sequences=10000):
        """
        Crée X, y pour time-series multi-output.

        X: (n_seq, seq_len, n_features)
        y: (n_seq, horizon, n_targets)
        """
        # 1) On travaille sur toutes les colonnes du DF pour les index
        all_cols = self.data.columns.tolist()

        # 2) Vérifier que toutes les cibles existent
        valid_targets = [t for t in targets if t in all_cols]
        if len(valid_targets) != len(targets):
            missing = [t for t in targets if t not in all_cols]
            raise ValueError(f"Cibles manquantes dans data: {missing}")

        # 3) Features: si 'features' ne contient pas les targets, on les ajoute
        #    pour avoir la même base de données, puis on utilisera leurs index indépendamment
        if not features:
            # fallback : toutes les colonnes sauf cibles
            features = [c for c in all_cols if c not in valid_targets]
        else:
            # s'assurer que les features existent
            features = [f for f in features if f in all_cols]

        # Liste finale de colonnes pour X (features + targets si pas déjà dedans)
        cols_for_X = features + [c for c in valid_targets if c not in features]

        data_array = self.data[cols_for_X].values

        # indices des targets dans ce sous-ensemble
        target_indices = [cols_for_X.index(t) for t in valid_targets]

        X, y = [], []
        n_samples = len(data_array)

        max_end = n_samples - self.sequence_length - self.prediction_horizon + 1
        if max_end <= 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Pas adaptatif pour limiter à max_sequences
        step = max(1, max_end // max_sequences)

        for i in range(0, max_end, step):
            # Toutes les colonnes pour X
            X.append(data_array[i:i + self.sequence_length])
            # Uniquement les colonnes cibles pour y
            y.append(
                data_array[
                    i + self.sequence_length:
                    i + self.sequence_length + self.prediction_horizon,
                    target_indices
                ]
            )
            if len(X) >= max_sequences:
                break

        X = np.array(X)
        y = np.array(y)

        # Split train/test
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, X_test, y_train, y_test

    # =====================================================================
    # 2) Baselines
    # =====================================================================

    def create_persistence_model(self):
        """Baseline Persistance: répète la dernière valeur sur tout l'horizon."""

        def persistence_predict(X, n_targets, horizon):
            last_values = X[:, -1, :n_targets]
            return np.tile(last_values[:, np.newaxis, :], (1, horizon, 1))

        return persistence_predict

    def create_moving_average_model(self, window=5):
        """Baseline Moyenne mobile sur les derniers pas."""

        def moving_average_predict(X, n_targets, horizon):
            last_values = X[:, -window:, :n_targets]
            avg = np.mean(last_values, axis=1)
            return np.tile(avg[:, np.newaxis, :], (1, horizon, 1))

        return moving_average_predict

    # =====================================================================
    # 3) ML
    # =====================================================================

    def create_linear_regression_model(self):
        return MultiOutputRegressor(LinearRegression())

    def create_random_forest_model(self, n_estimators=100):
        return MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1,
                max_depth=15
            )
        )

    def create_xgboost_model(self):
        return MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        )

    def create_gradient_boosting_model(self):
        return MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        )

    # =====================================================================
    # 4) Deep Learning – sorties (horizon * n_targets)
    # =====================================================================

    def _compile_dl_model(self, model, lr=0.001):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='mse',
            metrics=['mae']
        )
        return model

    def create_lstm_model(self, input_shape, n_targets):
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(n_targets * self.prediction_horizon)
        ])
        return self._compile_dl_model(model)

    def create_gru_model(self, input_shape, n_targets):
        model = keras.Sequential([
            layers.GRU(64, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.GRU(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(n_targets * self.prediction_horizon)
        ])
        return self._compile_dl_model(model)

    def create_bilstm_model(self, input_shape, n_targets):
        model = keras.Sequential([
            layers.Bidirectional(
                layers.LSTM(32, return_sequences=False),
                input_shape=input_shape
            ),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(n_targets * self.prediction_horizon)
        ])
        return self._compile_dl_model(model)

    def create_cnn_lstm_model(self, input_shape, n_targets):
        model = keras.Sequential([
            layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(n_targets * self.prediction_horizon)
        ])
        return self._compile_dl_model(model)

    def create_transformer_like_model(self, input_shape, n_targets):
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation='relu'),
            layers.Dense(n_targets * self.prediction_horizon)
        ])
        return self._compile_dl_model(model)

    def create_mlp_model(self, input_shape, n_targets):
        model = keras.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(n_targets * self.prediction_horizon)
        ])
        return self._compile_dl_model(model)

    # =====================================================================
    # 5) Entraînement + évaluation
    # =====================================================================

    def train_and_evaluate(
        self,
        model_name,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=30,
        batch_size=32,
        verbose=0
    ):
        """
        Entraîne un modèle (baseline, ML ou DL) et retourne:
        - objet modèle
        - dict {'MAE','RMSE','R2','Train_Time_s'}
        """
        start_time = time.time()
        n_targets = y_train.shape[2]
        input_shape = (X_train.shape[1], X_train.shape[2])

        # Baselines
        if model_name == 'Persistence (baseline)':
            model = self.create_persistence_model()
            y_pred = model(X_test, n_targets, self.prediction_horizon)

        elif model_name == 'Moving Average (baseline)':
            model = self.create_moving_average_model(window=min(5, self.sequence_length))
            y_pred = model(X_test, n_targets, self.prediction_horizon)

        # ML
        elif model_name == 'Linear Regression':
            model = self.create_linear_regression_model()
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            y_train_flat = y_train.reshape(y_train.shape[0], -1)
            model.fit(X_train_flat, y_train_flat)

            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            y_pred_flat = model.predict(X_test_flat)
            y_pred = y_pred_flat.reshape(-1, self.prediction_horizon, n_targets)

        elif model_name == 'Random Forest':
            model = self.create_random_forest_model()
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            y_train_flat = y_train.reshape(y_train.shape[0], -1)
            model.fit(X_train_flat, y_train_flat)

            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            y_pred_flat = model.predict(X_test_flat)
            y_pred = y_pred_flat.reshape(-1, self.prediction_horizon, n_targets)

        elif model_name == 'XGBoost':
            model = self.create_xgboost_model()
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            y_train_flat = y_train.reshape(y_train.shape[0], -1)
            model.fit(X_train_flat, y_train_flat)

            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            y_pred_flat = model.predict(X_test_flat)
            y_pred = y_pred_flat.reshape(-1, self.prediction_horizon, n_targets)

        elif model_name == 'Gradient Boosting':
            model = self.create_gradient_boosting_model()
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            y_train_flat = y_train.reshape(y_train.shape[0], -1)
            model.fit(X_train_flat, y_train_flat)

            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            y_pred_flat = model.predict(X_test_flat)
            y_pred = y_pred_flat.reshape(-1, self.prediction_horizon, n_targets)

        # DL
        else:
            if model_name == 'LSTM':
                model = self.create_lstm_model(input_shape, n_targets)
            elif model_name == 'GRU':
                model = self.create_gru_model(input_shape, n_targets)
            elif model_name == 'BiLSTM':
                model = self.create_bilstm_model(input_shape, n_targets)
            elif model_name == 'CNN_LSTM':
                model = self.create_cnn_lstm_model(input_shape, n_targets)
            elif model_name == 'Transformer':
                model = self.create_transformer_like_model(input_shape, n_targets)
            elif model_name == 'MLP':
                model = self.create_mlp_model(input_shape, n_targets)
            else:
                raise ValueError(f"Modèle DL non géré: {model_name}")

            y_train_flat = y_train.reshape(y_train.shape[0], -1)
            y_test_flat = y_test.reshape(y_test.shape[0], -1)

            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            history = model.fit(
                X_train, y_train_flat,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=[early_stop]
            )
            self.histories[model_name] = history

            y_pred_flat = model.predict(X_test, verbose=0)
            y_pred = y_pred_flat.reshape(-1, self.prediction_horizon, n_targets)

        train_time = time.time() - start_time

        mae = mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))
        rmse = np.sqrt(mean_squared_error(y_test.reshape(-1), y_pred.reshape(-1)))
        r2 = r2_score(y_test.reshape(-1), y_pred.reshape(-1))

        metrics = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'Train_Time_s': float(train_time),
            'Model': model_name
        }

        return model, y_pred, metrics
