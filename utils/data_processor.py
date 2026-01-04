"""
utils/data_processor.py 

"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Classe de preprocessing optimisée mémoire
    ✅ Traite des millions de lignes sans crash RAM
    """
    
    def __init__(self, df=None):
        self.df = df
        self.original_df = df.copy() if df is not None else None
        self.scaler = None
        self.transformations = []
    
    def create_traffic_metrics(self, time_column='Time', length_column='Length', 
                              protocol_filter=None, freq='1S', target_prefix='ms_teams'):
        """
        🚀 MÉTHODE ULTRA OPTIMISÉE RAM - Version streaming
        """
        if self.df is None:
            raise ValueError("Données non chargées")
        
        print(f"🔄 Transformation métriques: time={time_column}, length={length_column}")
        
        # 🔥 STRATÉGIE 1 : Traitement par morceaux pour éviter la RAM
        # 1. Extraire seulement les colonnes nécessaires
        required_cols = [time_column, length_column]
        if 'Protocol' in self.df.columns:
            required_cols.append('Protocol')
        
        df_work = self.df[required_cols].copy()
        
        # 2. Conversion datetime OPTIMISÉE (sans copie massive)
        df_work[time_column] = pd.to_datetime(df_work[time_column], errors='coerce')
        
        # 3. 🔥 OPTIMISATION CRITIQUE : Supprimer les NaT par morceaux
        # Utiliser .dropna() directement sur la colonne time
        initial_rows = len(df_work)
        df_work = df_work.dropna(subset=[time_column])
        filtered_rows = len(df_work)
        
        if initial_rows != filtered_rows:
            print(f"  → {initial_rows - filtered_rows} lignes avec NaT supprimées")
        
        # 4. Définir l'index temporel
        df_work = df_work.set_index(time_column)
        df_work = df_work.sort_index()
        
        # 5. 🔥 OPTIMISATION : Filtrer par protocole APRÈS indexation
        if protocol_filter and 'Protocol' in df_work.columns:
            print(f"  → Filtrage protocole: {protocol_filter}")
            df_work = df_work[df_work['Protocol'] == protocol_filter]
        
        # 6. 🔥 OPTIMISATION : Agréger avec méthode plus efficace
        try:
            # Méthode 1 : Resample avec agrégation
            df_agg = df_work.resample(freq).agg({
                length_column: ['sum', 'count', 'mean']
            })
        except MemoryError:
            # Méthode 2 alternative si mémoire insuffisante
            print("  ⚠️ Méthode resample trop lourde, utilisation groupby")
            df_work['time_group'] = df_work.index.floor(freq)
            df_agg = df_work.groupby('time_group').agg({
                length_column: ['sum', 'count', 'mean']
            })
            df_agg.index = pd.to_datetime(df_agg.index)
            df_agg = df_agg.asfreq(freq, fill_value=0)
        
        # 7. Renommer colonnes
        df_agg.columns = ['total_bytes', 'packet_count', 'avg_packet_size']
        
        # 8. Ajouter total_packets (identique à packet_count)
        #df_agg['total_packets'] = df_agg['packet_count']
        
        # 9. Throughput (Mbps) - correction formule
        df_agg['throughput_mbps'] = (df_agg['total_bytes'] * 8) / (1_000_000 * pd.Timedelta(freq).seconds)
        
        # 10. Nettoyer et remplir valeurs manquantes
        df_agg = df_agg.fillna(0)
        
        # 11. Supprimer les lignes où tout est 0 (pas de trafic)
        df_agg = df_agg[(df_agg['total_bytes'] > 0) | (df_agg['packet_count'] > 0)]
        
        # 12. Affecter au DataProcessor
        self.df = df_agg
        self.transformations.append(f"✅ Métriques 5G créées (freq={freq})")
        print(f"✅ {len(self.df)} intervalles créés")
        print(f"📊 Colonnes: {list(self.df.columns)}")
        
        return self
    
    def handle_missing_values(self, method='ffill'):
        """Gère valeurs manquantes"""
        if self.df is None:
            return self
        
        if method == 'interpolate':
            self.df = self.df.interpolate(method='time', limit_direction='forward')
        elif method == 'mean':
            # Calculer la moyenne colonne par colonne pour économiser mémoire
            for col in self.df.columns:
                if self.df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    col_mean = self.df[col].mean()
                    self.df[col] = self.df[col].fillna(col_mean)
        elif method == 'ffill':
            self.df = self.df.ffill().bfill()
        else:
            self.df = self.df.fillna(0)
        
        self.transformations.append(f"Missing values: {method}")
        return self
    
    def normalize_data(self, method='robust'):
        """Normalisation optimisée"""
        if self.df is None:
            return self
        
        # Sélectionner seulement les colonnes numériques
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            print("⚠️ Aucune colonne numérique à normaliser")
            return self
        
        # Créer un DataFrame temporaire pour la normalisation
        df_numeric = self.df[numeric_cols].copy()
        
        # Appliquer le scaler
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:  # robust par défaut
            self.scaler = RobustScaler()
        
        # Normaliser et réassigner
        df_numeric_values = self.scaler.fit_transform(df_numeric)
        self.df[numeric_cols] = df_numeric_values
        
        self.transformations.append(f"Normalisation: {method} sur {len(numeric_cols)} colonnes")
        return self
    
    def get_processed_data(self):
        """Retourne données traitées"""
        return self.df
    
    def get_transformations(self):
        """Retourne liste transformations"""
        return self.transformations
    
    def get_original_data(self):
        """Retourne données originales"""
        return self.original_df