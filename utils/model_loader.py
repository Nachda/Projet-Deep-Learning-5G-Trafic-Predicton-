"""
utils/model_loader.py 

"""
import joblib
import tensorflow as tf
import os
import json

# ✅ FIX: detected_models 
MODELS_DIR = './models'
detected_models = [
    # Keras models
    {'name': 'lstm_v1', 'file': 'lstm_model.keras', 'type': 'KERAS'},
    {'name': 'gru_v1', 'file': 'gru_model.keras', 'type': 'KERAS'},
    {'name': 'bilstm_v1', 'file': 'bilstm_model.keras', 'type': 'KERAS'},
    {'name': 'transformer_v1', 'file': 'transformer_model.keras', 'type': 'KERAS'},
    # Scikit-learn models
    {'name': 'xgboost_v1', 'file': 'xgboost_model.pkl', 'type': 'PKL'},
    {'name': 'random_forest_v1', 'file': 'rf_model.pkl', 'type': 'PKL'},
    {'name': 'gradient_boosting_v1', 'file': 'gb_model.pkl', 'type': 'PKL'},
    {'name': 'linear_regression_v1', 'file': 'lr_model.pkl', 'type': 'PKL'},
]

class ModelLoader:
    """Charge TOUS les modèles détectés automatiquement"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.metadata = {}
    
    def load_all(self):
        """Charge TOUS tes modèles (.pkl + .keras)"""
        print("\n🔄 **CHARGEMENT AUTOMATIQUE MODÈLES**")
        
        # 1. Scalers
        for scaler_file in ['scaler.pkl', 'scaler_multi_output.pkl']:
            scaler_path = os.path.join(MODELS_DIR, scaler_file)
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Scaler: {scaler_file}")
        
        # 2. Metadata
        metadata_path = os.path.join(MODELS_DIR, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"✅ Metadata chargée")
        
        # 3. TOUS les .pkl
        for model_info in detected_models:
            if model_info['type'] == 'PKL':
                try:
                    model_path = os.path.join(MODELS_DIR, model_info['file'])
                    if os.path.exists(model_path):
                        model_name = model_info['name']
                        self.models[model_name] = joblib.load(model_path)
                        print(f"✅ {model_name} (.pkl)")
                except Exception as e:
                    print(f"⚠️ Erreur {model_info['file']}: {e}")
        
        # 4. TOUS les .keras
        for model_info in detected_models:
            if model_info['type'] == 'KERAS':
                try:
                    model_path = os.path.join(MODELS_DIR, model_info['file'])
                    if os.path.exists(model_path):
                        model_name = model_info['name']
                        self.models[model_name] = tf.keras.models.load_model(model_path)
                        print(f"✅ {model_name} (.keras)")
                except Exception as e:
                    print(f"⚠️ Erreur {model_info['file']}: {e}")
        
        print(f"\n🎉 **TOTAL: {len(self.models)} modèles chargés**")
        return self.models, self.scaler, self.metadata

# Test automatique
if __name__ == "__main__":
    loader = ModelLoader()
    models, scaler, metadata = loader.load_all()
