# utils/report_generator.py
import pandas as pd
import numpy as np
from datetime import datetime
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import plotly.graph_objects as go


class ReportGenerator:
    """Classe pour la génération de rapports professionnels"""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles = {}

    def create_pdf_report(self, data, output_path="report.pdf"):
        """Crée un rapport PDF professionnel"""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )

        elements = []

        # Titre
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1E3A8A')
        )
        elements.append(Paragraph("Rapport d'Analyse Prédictive 5G", title_style))

        # Métadonnées
        meta_text = f"""
        <b>Date de génération:</b> {datetime.now().strftime('%d %B %Y à %H:%M')}<br/>
        <b>Système:</b> 5G Traffic Predictor Pro<br/>
        <b>Version:</b> 2.0.0
        """
        elements.append(Paragraph(meta_text, self.styles["Normal"]))
        elements.append(Spacer(1, 20))

        # Résumé exécutif
        elements.append(Paragraph("Résumé Exécutif", self.styles["Heading2"]))

        summary_text = """
        Ce rapport présente une analyse complète du trafic réseau 5G basée sur des modèles 
        prédictifs avancés. Les analyses incluent l'exploration des données, l'entraînement 
        de multiples modèles, les prédictions multi-horizon et les recommandations d'actions 
        réseau.
        """
        elements.append(Paragraph(summary_text, self.styles["Normal"]))
        elements.append(Spacer(1, 20))

        # Points clés
        elements.append(Paragraph("Points Clés", self.styles["Heading3"]))

        key_points = [
            "Analyse basée sur les données MS Teams 5G",
            "Comparaison de 15+ modèles prédictifs",
            "Prédictions multi-horizon jusqu'à 60 secondes",
            "Recommandations d'actions réseau automatisées",
            "Score de santé réseau en temps réel"
        ]

        for point in key_points:
            elements.append(Paragraph(f"• {point}", self.styles["Normal"]))

        elements.append(Spacer(1, 20))

        # Performance des modèles
        if 'model_performance' in data:
            elements.append(Paragraph("Performance des Modèles", self.styles["Heading2"]))

            model_data = [['Modèle', 'MAE', 'R²', 'Temps (s)']]

            for model in data['model_performance']:
                # compatibilité: Train_Time_s ou Training_Time
                time_val = model.get('Train_Time_s', model.get('Training_Time', 0))
                model_data.append([
                    model['Model'],
                    f"{model['MAE']:.4f}",
                    f"{model['R2']:.4f}",
                    f"{time_val:.1f}"
                ])

            table = Table(model_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            elements.append(table)
            elements.append(Spacer(1, 20))

        # Recommandations
        elements.append(Paragraph("Recommandations", self.styles["Heading2"]))

        recommendations = [
            ("Optimisation bande passante", "Ajuster l'allocation selon les prédictions de trafic"),
            ("Surveillance proactive", "Mettre en place des alertes basées sur les tendances"),
            ("Maintenance préventive", "Planifier les interventions pendant les périodes creuses"),
            ("Amélioration modèles", "Ré-entraîner régulièrement avec de nouvelles données")
        ]

        for title, description in recommendations:
            elements.append(Paragraph(f"<b>{title}</b>", self.styles["Normal"]))
            elements.append(Paragraph(description, self.styles["Normal"]))
            elements.append(Spacer(1, 10))

        # Pied de page
        footer_text = """
        <i>Rapport généré automatiquement par 5G Traffic Predictor Pro. 
        Pour plus d'informations, contactez support@5gpredictor.com</i>
        """
        elements.append(Paragraph(footer_text, self.styles["Italic"]))

        # Générer le PDF
        doc.build(elements)

        return output_path

    def create_excel_report(self, data, output_path="report.xlsx"):
        """Crée un rapport Excel avec plusieurs onglets"""
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Onglet Résumé
            summary_data = {
                'Métrique': [
                    'Date de génération',
                    'Nombre de modèles',
                    'Meilleur modèle',
                    'MAE moyen',
                    'R² moyen',
                    'Score santé réseau'
                ],
                'Valeur': [
                    datetime.now().strftime('%Y-%m-%d %H:%M'),
                    len(data.get('model_performance', [])),
                    data.get('best_model', 'N/A'),
                    f"{data.get('avg_mae', 0):.4f}",
                    f"{data.get('avg_r2', 0):.4f}",
                    f"{data.get('health_score', 0):.1f}/100"
                ]
            }

            pd.DataFrame(summary_data).to_excel(
                writer, sheet_name='Résumé', index=False
            )

            # Onglet Performance
            if 'model_performance' in data:
                df_performance = pd.DataFrame(data['model_performance'])
                pd.DataFrame(df_performance).to_excel(
                    writer, sheet_name='Performance Modèles', index=False
                )

            # Onglet Prédictions
            if 'predictions' in data:
                pred_data = []
                for model_name, predictions in data['predictions'].items():
                    for i, pred in enumerate(predictions):
                        pred_data.append({
                            'Modèle': model_name,
                            'Horizon': i + 1,
                            'Throughput': pred[0],
                            'Packets': pred[1]
                        })

                pd.DataFrame(pred_data).to_excel(
                    writer, sheet_name='Prédictions', index=False
                )

            # Onglet Actions
            if 'actions' in data:
                pd.DataFrame(data['actions']).to_excel(
                    writer, sheet_name='Actions', index=False
                )

        return output_path

    def create_html_report(self, data, output_path="report.html"):
        """Crée un rapport HTML interactif"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Rapport 5G Traffic Predictor Pro</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 10px 0;
                    border-left: 5px solid #667eea;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #667eea;
                    color: white;
                }}
                .alert {{
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                .alert-success {{
                    background-color: #d4edda;
                    color: #155724;
                    border: 1px solid #c3e6cb;
                }}
                .alert-warning {{
                    background-color: #fff3cd;
                    color: #856404;
                    border: 1px solid #ffeaa7;
                }}
                .alert-danger {{
                    background-color: #f8d7da;
                    color: #721c24;
                    border: 1px solid #f5c6cb;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📡 Rapport d'Analyse Prédictive 5G</h1>
                    <p>Généré le {datetime.now().strftime('%d %B %Y à %H:%M')}</p>
                </div>
                
                <h2>📋 Résumé Exécutif</h2>
                <div class="metric-card">
        """

        # Ajouter les métriques principales
        if 'model_performance' in data and data['model_performance']:
            best_model = min(data['model_performance'], key=lambda x: x['MAE'])
            html_content += f"""
                    <p><strong>Modèle le plus performant:</strong> {best_model['Model']}</p>
                    <p><strong>MAE du meilleur modèle:</strong> {best_model['MAE']:.4f}</p>
                    <p><strong>R² du meilleur modèle:</strong> {best_model['R2']:.4f}</p>
            """

        # Score santé si dispo
        html_content += f"""
                    <p><strong>Score santé réseau:</strong> {data.get('health_score', 0):.1f}/100</p>
                </div>
        """

        # Tableau de performance
        if 'model_performance' in data and data['model_performance']:
            html_content += """
                <h2>📊 Performance des Modèles</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Modèle</th>
                            <th>MAE</th>
                            <th>R²</th>
                            <th>Temps d'entraînement (s)</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for model in data['model_performance']:
                time_val = model.get('Train_Time_s', model.get('Training_Time', 0))
                html_content += f"""
                        <tr>
                            <td>{model['Model']}</td>
                            <td>{model['MAE']:.4f}</td>
                            <td>{model['R2']:.4f}</td>
                            <td>{time_val:.1f}</td>
                        </tr>
                """

            html_content += """
                    </tbody>
                </table>
            """

        # Recommandations
        html_content += """
                <h2>🎯 Recommandations</h2>
                <div class="alert alert-success">
                    <strong>✅ Optimisation recommandée:</strong> Mettre à jour les modèles prédictifs hebdomadairement
                </div>
                <div class="alert alert-warning">
                    <strong>⚠️ Surveillance requise:</strong> Surveiller les pics de trafic entre 18h et 20h
                </div>
                <div class="alert alert-danger">
                    <strong>🚨 Action immédiate:</strong> Vérifier la configuration QoS pour les flux vidéo
                </div>
                
                <hr>
                <footer>
                    <p><em>Rapport généré par 5G Traffic Predictor Pro v2.0.0</em></p>
                    <p><em>Contact: support@5gpredictor.com | Documentation: docs.5gpredictor.com</em></p>
                </footer>
            </div>
        </body>
        </html>
        """

        # Sauvegarder
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def generate_comprehensive_report(self, data, formats=['pdf', 'excel', 'html']):
        """Génère un rapport complet dans plusieurs formats"""
        reports = {}

        for fmt in formats:
            if fmt == 'pdf':
                reports['pdf'] = self.create_pdf_report(data)
            elif fmt == 'excel':
                reports['excel'] = self.create_excel_report(data)
            elif fmt == 'html':
                reports['html'] = self.create_html_report(data)

        return reports

    @staticmethod
    def plot_to_base64(fig, format='png'):
        """Convertit un graphique Plotly en base64"""
        buffer = BytesIO()
        fig.write_image(buffer, format=format)
        buffer.seek(0)

        img_str = base64.b64encode(buffer.read()).decode()
        return f"data:image/{format};base64,{img_str}"
