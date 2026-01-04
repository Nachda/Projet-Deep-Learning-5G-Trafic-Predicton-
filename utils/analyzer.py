# analyzer.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ModelAnalyzer:
    """Classe pour l'analyse des modèles et prédictions"""
    
    def __init__(self, model_results=None):
        self.model_results = model_results
    
    def calculate_composite_score(self, df_results, weights=None):
        if weights is None:
            weights = {
                'MAE': 0.4,          # plus bas = mieux
                'R2': 0.4,           # plus haut = mieux
                'Train_Time_s': 0.2  # plus bas = mieux
            }

        df = df_results.copy()

        for metric, weight in weights.items():
            if metric in df.columns:
                if metric == 'R2':
                    df[f'{metric}_norm'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
                else:
                    df[f'{metric}_norm'] = 1 - (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())

        df['Composite_Score'] = 0
        for metric, weight in weights.items():
            if f'{metric}_norm' in df.columns:
                df['Composite_Score'] += df[f'{metric}_norm'] * weight

        df['Composite_Score'] = df['Composite_Score'] * 100
        return df
    
    def analyze_error_distribution(self, y_true, y_pred):
        """
        Analyse la distribution des erreurs
        
        Returns:
            dict: Statistiques sur les erreurs
        """
        errors = y_pred - y_true
        
        analysis = {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'mae': float(np.mean(np.abs(errors))),
            'max_abs_error': float(np.max(np.abs(errors))),
            'error_skewness': float(stats.skew(errors.flatten())),
            'error_kurtosis': float(stats.kurtosis(errors.flatten()))
        }
        
        # Test de normalité
        try:
            shapiro_test = stats.shapiro(errors.flatten())
            analysis['shapiro_statistic'] = float(shapiro_test.statistic)
            analysis['shapiro_pvalue'] = float(shapiro_test.pvalue)
            analysis['is_normal'] = shapiro_test.pvalue > 0.05
        except:
            analysis['shapiro_statistic'] = None
            analysis['shapiro_pvalue'] = None
            analysis['is_normal'] = None
        
        return analysis
    
    def calculate_horizon_metrics(self, y_true, y_pred):
        """
        Calcule les métriques par horizon de prédiction
        
        Returns:
            DataFrame: Métriques par pas de temps
        """
        horizon = y_true.shape[1]
        
        metrics_by_horizon = []
        
        for h in range(horizon):
            y_true_h = y_true[:, h, :]
            y_pred_h = y_pred[:, h, :]
            
            mae = np.mean(np.abs(y_pred_h - y_true_h))
            rmse = np.sqrt(np.mean((y_pred_h - y_true_h) ** 2))
            
            # R² pour chaque feature
            r2_values = []
            for f in range(y_true_h.shape[1]):
                try:
                    from sklearn.metrics import r2_score
                    r2 = r2_score(y_true_h[:, f], y_pred_h[:, f])
                    r2_values.append(r2)
                except:
                    r2_values.append(np.nan)
            
            metrics_by_horizon.append({
                'Horizon': h + 1,
                'MAE': float(mae),
                'RMSE': float(rmse),
                'R2_Avg': float(np.nanmean(r2_values)),
                'R2_Min': float(np.nanmin(r2_values)),
                'R2_Max': float(np.nanmax(r2_values))
            })
        
        return pd.DataFrame(metrics_by_horizon)
    
    def detect_anomalies(self, predictions, threshold_std=3):
        """
        Détecte les anomalies dans les prédictions
        
        Returns:
            dict: Anomalies détectées
        """
        # Calculer la moyenne et écart-type sur tous les modèles
        preds_array = np.array(list(predictions.values()))
        mean_pred = np.mean(preds_array, axis=0)
        std_pred = np.std(preds_array, axis=0)
        
        # Détecter les anomalies (valeurs > N écarts-types de la moyenne)
        anomalies = np.abs(preds_array - mean_pred) > threshold_std * std_pred
        
        anomaly_info = {
            'total_anomalies': int(np.sum(anomalies)),
            'anomaly_rate': float(np.mean(anomalies)),
            'models_with_anomalies': [],
            'horizons_with_anomalies': []
        }
        
        # Détails par modèle
        for idx, model_name in enumerate(predictions.keys()):
            model_anomalies = np.sum(anomalies[idx])
            if model_anomalies > 0:
                anomaly_info['models_with_anomalies'].append({
                    'model': model_name,
                    'anomalies': int(model_anomalies),
                    'rate': float(model_anomalies / anomalies[idx].size)
                })
        
        # Détails par horizon
        horizon_anomalies = np.sum(anomalies, axis=(0,1,2))  # [models,horizon,features]
        for h, count in enumerate(horizon_anomalies):
            if count > 0:
                anomaly_info['horizons_with_anomalies'].append({
                    'horizon': int(h + 1),
                    'anomalies': int(count)
                })
        
        return anomaly_info
    
    def compare_models_statistically(self, predictions_dict, y_true=None):
        """
        Compare statistiquement les modèles
        
        Returns:
            dict: Résultats des tests statistiques
        """
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)
        
        if n_models < 2:
            return {"error": "Au moins 2 modèles requis pour la comparaison"}
        
        # Préparer les données
        all_errors = []
        for model_name, pred in predictions_dict.items():
            if y_true is not None and pred.shape == y_true.shape:
                errors = np.abs(pred - y_true).flatten()
            else:
                # Si pas de y_true, utiliser la variance comme proxy
                errors = np.std(pred, axis=0).flatten()
            
            all_errors.append(errors)
        
        # Test ANOVA (si assez de données)
        if len(all_errors[0]) > 30:
            try:
                f_stat, p_value = stats.f_oneway(*all_errors)
                anova_result = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
            except:
                anova_result = None
        else:
            anova_result = None
        
        # Tests de comparaison par paire
        pairwise_comparisons = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                try:
                    # Test t de Student
                    t_stat, t_pvalue = stats.ttest_ind(all_errors[i], all_errors[j])
                    
                    # Test de Wilcoxon (non-paramétrique)
                    w_stat, w_pvalue = stats.ranksums(all_errors[i], all_errors[j])
                    
                    pairwise_comparisons.append({
                        'model1': model_names[i],
                        'model2': model_names[j],
                        't_statistic': float(t_stat),
                        't_pvalue': float(t_pvalue),
                        't_significant': t_pvalue < 0.05,
                        'w_statistic': float(w_stat),
                        'w_pvalue': float(w_pvalue),
                        'w_significant': w_pvalue < 0.05,
                        'mean_diff': float(np.mean(all_errors[i]) - np.mean(all_errors[j]))
                    })
                except:
                    continue
        
        return {
            'anova': anova_result,
            'pairwise_comparisons': pairwise_comparisons,
            'n_models': n_models
        }

class NetworkOptimizer:
    """Classe pour l'optimisation des actions réseau"""
    
    def __init__(self, current_state, predictions):
        self.current_state = current_state
        self.predictions = predictions
    
    def analyze_network_health(self):
        """Analyse la santé globale du réseau"""
        if not self.predictions:
            return None
        
        # Utiliser les prédictions du modèle principal
        primary_model = list(self.predictions.keys())[0]
        preds = self.predictions[primary_model]
        
        # Métriques de santé
        throughput = preds[:, 0]
        packets = preds[:, 1]
        
        health_metrics = {
            'throughput_avg': float(np.mean(throughput)),
            'throughput_max': float(np.max(throughput)),
            'throughput_std': float(np.std(throughput)),
            'packets_avg': float(np.mean(packets)),
            'packets_variance': float(np.var(packets)),
            'stability_score': self._calculate_stability_score(throughput),
            'capacity_utilization': self._calculate_capacity_utilization(throughput),
            'trend_direction': self._calculate_trend(throughput)
        }
        
        # Score de santé global
        health_score = self._calculate_health_score(health_metrics)
        health_metrics['health_score'] = health_score
        health_metrics['health_level'] = self._get_health_level(health_score)
        
        return health_metrics
    
    def _calculate_stability_score(self, throughput):
        """Calcule un score de stabilité"""
        # Coefficient de variation
        cv = np.std(throughput) / np.mean(throughput)
        
        # Convertir en score 0-100
        stability = max(0, 100 - cv * 1000)
        return min(100, stability)
    
    def _calculate_capacity_utilization(self, throughput, max_capacity=5.0):
        """Calcule l'utilisation de la capacité"""
        avg_throughput = np.mean(throughput)
        utilization = (avg_throughput / max_capacity) * 100
        return min(100, utilization)
    
    def _calculate_trend(self, throughput):
        """Calcule la tendance"""
        if len(throughput) < 2:
            return 0
        
        # Régression linéaire simple
        x = np.arange(len(throughput))
        slope, _ = np.polyfit(x, throughput, 1)
        
        return float(slope)
    
    def _calculate_health_score(self, metrics):
        """Calcule un score de santé global"""
        weights = {
            'stability_score': 0.3,
            'capacity_utilization': 0.3,
            'throughput_std': 0.2,
            'trend_direction': 0.2
        }
        
        score = 0
        
        # Contribution de la stabilité
        stability = metrics['stability_score']
        score += stability * weights['stability_score']
        
        # Contribution de l'utilisation (inversée: moins d'utilisation = meilleur)
        utilization = metrics['capacity_utilization']
        score += (100 - utilization) * weights['capacity_utilization']
        
        # Contribution de la variance (inversée)
        std_score = max(0, 100 - metrics['throughput_std'] * 100)
        score += std_score * weights['throughput_std']
        
        # Contribution de la tendance
        trend = metrics['trend_direction']
        if trend < -0.1:  # Forte baisse
            trend_score = 30
        elif trend < 0:    # Légère baisse
            trend_score = 70
        elif trend > 0.1:  # Forte hausse
            trend_score = 80
        else:              # Stable
            trend_score = 100
        
        score += trend_score * weights['trend_direction']
        
        return min(100, score)
    
    def _get_health_level(self, score):
        """Détermine le niveau de santé"""
        if score >= 80:
            return 'Excellent'
        elif score >= 60:
            return 'Bon'
        elif score >= 40:
            return 'Modéré'
        else:
            return 'Critique'
    
    def generate_actions(self, health_metrics):
        """Génère des actions recommandées basées sur la santé du réseau"""
        actions = []
        
        # Actions pour haute utilisation
        if health_metrics['capacity_utilization'] > 85:
            actions.append({
                'type': 'critical',
                'action': 'Augmenter bande passante',
                'priority': 1,
                'parameters': {'increase': '25%', 'timeframe': 'immédiat'},
                'expected_impact': 'Réduction congestion, amélioration QoS',
                'cost': 'élevé'
            })
        
        # Actions pour instabilité
        if health_metrics['stability_score'] < 60:
            actions.append({
                'type': 'warning',
                'action': 'Optimiser paramètres RF',
                'priority': 2,
                'parameters': {'power_adjustment': '+3dB', 'handover_optimization': True},
                'expected_impact': 'Amélioration stabilité signal',
                'cost': 'moyen'
            })
        
        # Actions pour tendance négative
        if health_metrics['trend_direction'] < -0.2:
            actions.append({
                'type': 'warning',
                'action': 'Analyser causes de dégradation',
                'priority': 2,
                'parameters': {'monitoring_period': '24h', 'root_cause_analysis': True},
                'expected_impact': 'Identification problèmes sous-jacents',
                'cost': 'faible'
            })
        
        # Actions préventives générales
        actions.append({
            'type': 'maintenance',
            'action': 'Mettre à jour modèles prédictifs',
            'priority': 3,
            'parameters': {'retrain': True, 'data_window': '7 jours'},
            'expected_impact': 'Amélioration précision prédictions',
            'cost': 'faible'
        })
        
        # Trier par priorité
        actions.sort(key=lambda x: x['priority'])
        
        return actions