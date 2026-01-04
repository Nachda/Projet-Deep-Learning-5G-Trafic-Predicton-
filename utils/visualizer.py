# visualizer.py 
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
from scipy import stats

class DataVisualizer:
    @staticmethod
    def plot_time_series(df, columns=None, title="Séries Temporelles"):
        if columns is None:
            columns = df.columns.tolist()[:4]
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        
        for idx, col in enumerate(columns):
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], mode='lines', name=col,
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))
        
        fig.update_layout(title=title, xaxis_title="Temps", yaxis_title="Valeur", 
                         height=500, template="plotly_white", hovermode='x unified')
        return fig

class TrainingVisualizer:
    """Classe pour la visualisation de l'entraînement"""
    
    @staticmethod
    def plot_training_history(history, title="Historique d'Entraînement"):
        """Affiche l'historique d'entraînement d'un modèle DL"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Loss', 'MAE')
        )
        
        # Loss
        fig.add_trace(
            go.Scatter(
                y=history.history['loss'],
                mode='lines',
                name='Train Loss',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        if 'val_loss' in history.history:
            fig.add_trace(
                go.Scatter(
                    y=history.history['val_loss'],
                    mode='lines',
                    name='Val Loss',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
        
        # MAE
        if 'mae' in history.history:
            fig.add_trace(
                go.Scatter(
                    y=history.history['mae'],
                    mode='lines',
                    name='Train MAE',
                    line=dict(color='green')
                ),
                row=1, col=2
            )
        
        if 'val_mae' in history.history:
            fig.add_trace(
                go.Scatter(
                    y=history.history['val_mae'],
                    mode='lines',
                    name='Val MAE',
                    line=dict(color='orange')
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title=title,
            height=400,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Epochs", row=1, col=1)
        fig.update_xaxes(title_text="Epochs", row=1, col=2)
        
        return fig
    
    @staticmethod
    def plot_model_comparison(df_results, metric='MAE', title="Comparaison des Modèles"):
        """Affiche la comparaison des modèles"""
        fig = px.bar(
            df_results,
            x='Model',
            y=metric,
            color=metric,
            color_continuous_scale='RdYlGn_r' if metric == 'MAE' else 'RdYlGn',
            title=title
        )
        
        fig.update_layout(
            height=500,
            template="plotly_white",
            xaxis_tickangle=-45
        )
        
        return fig

class PredictionVisualizer:
    @staticmethod
    def plot_prediction_errors(y_true, y_pred, title="Analyse des Erreurs"):
        """✅ FIX: QQ-plot simplifié SANS statsmodels"""
        errors = y_pred - y_true
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            'Distribution des erreurs', 'Erreurs vs Valeurs prédites',
            'Autocorrélation', 'QQ-Plot simplifié'
        ))
        
        # Histogramme
        fig.add_trace(go.Histogram(x=errors.flatten(), nbinsx=50, name='Distribution', marker_color='blue'), 1, 1)
        
        # Erreurs vs prédites
        fig.add_trace(go.Scatter(x=y_pred.flatten(), y=errors.flatten(), mode='markers', 
                                marker=dict(size=3, color='red')), 1, 2)
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)
        
        # Autocorrélation simple
        lags = min(20, len(errors.flatten())//10)
        acf_vals = [np.corrcoef(errors.flatten()[i:], errors.flatten()[:len(errors.flatten())-i])[0,1] 
                   for i in range(lags)]
        fig.add_trace(go.Bar(x=list(range(lags)), y=acf_vals, name='Autocorrélation', marker_color='green'), 2, 1)
        fig.add_hline(y=0.2, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-0.2, line_dash="dash", line_color="red", row=2, col=1)
        
        # ✅ FIX QQ-Plot: Quantiles théoriques manuels
        sorted_errors = np.sort(errors.flatten())
        n = len(sorted_errors)
        theoretical_quantiles = stats.norm.ppf((np.arange(1, n+1) - 0.5) / n)
        sample_quantiles = sorted_errors
        
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sample_quantiles, 
                                mode='markers', name='QQ', marker_color='purple'), 2, 2)
        fig.add_trace(go.Scatter(x=[sorted_errors.min()*1.2, sorted_errors.max()*1.2], 
                                y=[sorted_errors.min()*1.2, sorted_errors.max()*1.2],
                                mode='lines', name='Ligne parfaite', line=dict(color='red')), 2, 2)
        
        fig.update_layout(title=title, height=700, template="plotly_white", showlegend=False)
        return fig

class ComparisonVisualizer:
    """Classe pour la visualisation comparative"""
    
    @staticmethod
    def plot_radar_chart(df_metrics, metrics, title="Radar Chart des Performances"):
        """Affiche un radar chart des performances"""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        for idx, (_, row) in enumerate(df_metrics.iterrows()):
            values = []
            for metric in metrics:
                if metric in row.index:
                    # Normaliser entre 0 et 1
                    val_min = df_metrics[metric].min()
                    val_max = df_metrics[metric].max()
                    if val_max > val_min:
                        if metric == 'MAE':  # Inverser pour MAE
                            norm_val = 1 - (row[metric] - val_min) / (val_max - val_min)
                        else:
                            norm_val = (row[metric] - val_min) / (val_max - val_min)
                    else:
                        norm_val = 0.5
                    
                    values.append(norm_val)
                else:
                    values.append(0)
            
            # Fermer le polygone
            values = values + [values[0]]
            metrics_closed = metrics + [metrics[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_closed,
                name=row['Model'],
                fill='toself',
                line=dict(color=colors[idx % len(colors)], width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=title,
            height=600,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def plot_performance_matrix(df_results, title="Matrice de Performance"):
        """Affiche une matrice de performance des modèles"""
        # Préparer les données
        metrics = ['MAE', 'R2', 'Training_Time']
        
        # Normaliser
        df_normalized = df_results.copy()
        for metric in metrics:
            if metric in df_normalized.columns:
                if metric == 'MAE':
                    df_normalized[f'{metric}_norm'] = 1 - (
                        (df_normalized[metric] - df_normalized[metric].min()) / 
                        (df_normalized[metric].max() - df_normalized[metric].min())
                    )
                else:
                    df_normalized[f'{metric}_norm'] = (
                        (df_normalized[metric] - df_normalized[metric].min()) / 
                        (df_normalized[metric].max() - df_normalized[metric].min())
                    )
        
        # Créer la heatmap
        heatmap_data = []
        for metric in metrics:
            if f'{metric}_norm' in df_normalized.columns:
                heatmap_data.append(df_normalized[f'{metric}_norm'].values)
        
        if heatmap_data:
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=df_normalized['Model'].values,
                y=[m.replace('_norm', '') for m in metrics if f'{m}_norm' in df_normalized.columns],
                colorscale='RdYlGn',
                text=np.round(np.array(heatmap_data), 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title=title,
                height=400,
                template="plotly_white",
                xaxis_tickangle=-45
            )
            
            return fig
        
        return None