# pages/6_📋_Reports_Export.py
import streamlit as st
from pathlib import Path
import sys
import pandas as pd

from styles import inject_global_styles, page_header
inject_global_styles()
page_header("📋 Rapports & Export", "Étape 6/6 - Dossiers professionnels 5G")

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'utils'))
sys.path.insert(0, str(BASE_DIR))


from utils.report_generator import ReportGenerator

st.title("📋 **6. Rapports & Export**")
st.markdown("**Rapports professionnels automatiques**")

# ====== VÉRIFICATIONS ======
if 'models_trained' not in st.session_state or not st.session_state.get('models_trained', False):
    st.warning("⚠️ **Pages 1+2 → Entraîne les modèles d'abord**")
    st.stop()

# ====== COLLECTE DONNÉES SESSION ======
def collect_session_data():
    """Collecte toutes les données de session pour rapport"""
    data = {
        'model_performance': st.session_state.get('model_results', pd.DataFrame()).to_dict('records'),
        'best_model': None,
        'avg_mae': 0.0,
        'avg_r2': 0.0,
        'health_score': 75.0  # Score par défaut, peut être mis à jour depuis page 5
    }

    # Meilleur modèle
    if 'model_results' in st.session_state and not st.session_state.model_results.empty:
        results_df = st.session_state.model_results
        # Si Composite_Score existe (page 4), l'utiliser
        if 'Composite_Score' in results_df.columns:
            best = results_df.sort_values('Composite_Score', ascending=False).iloc[0]
        else:
            best = results_df.sort_values('MAE').iloc[0]

        data['best_model'] = best['Model']
        data['avg_mae'] = float(results_df['MAE'].mean())
        data['avg_r2'] = float(results_df['R2'].mean())

    # Health score si stocké depuis page 5
    if 'network_health_score' in st.session_state:
        data['health_score'] = float(st.session_state.network_health_score)

    # Prédictions (si disponibles)
    if 'X_test' in st.session_state and 'trained_models' in st.session_state:
        data['predictions'] = {}
        # (Tu peux ajouter ici une logique de sample de prédictions si besoin)

    # Actions recommandées (fixes ou dynamiques)
    data['actions'] = [
        {'type': 'info', 'action': 'Réentraîner modèles hebdomadairement', 'priority': 1},
        {'type': 'warning', 'action': 'Surveiller pics 18h-20h', 'priority': 2},
        {'type': 'success', 'action': 'Configuration QoS optimale', 'priority': 3}
    ]

    return data

# ====== GÉNÉRATION RAPPORT ======
st.subheader("📄 **Génération Rapport**")

col1, col2, col3 = st.columns(3)

format_choice = col1.selectbox(
    "Format rapport",
    options=["HTML", "PDF", "Excel"],
    index=0
)

include_details = col2.checkbox("Inclure détails techniques", value=True)
include_graphs = col3.checkbox("Inclure graphiques", value=True)

if st.button("📄 **Générer Rapport Complet**", type="primary"):
    with st.spinner(f"📊 Génération rapport {format_choice}..."):
        try:
            generator = ReportGenerator()
            report_data = collect_session_data()

            if format_choice == "HTML":
                html_report = generator.create_html_report(report_data)
                st.session_state.full_report = html_report
                st.success("✅ **Rapport HTML généré !**")

                # Aperçu
                st.subheader("👀 **Aperçu Rapport HTML**")
                st.components.v1.html(html_report, height=800, scrolling=True)

            elif format_choice == "PDF":
                pdf_path = generator.create_pdf_report(report_data)
                st.success(f"✅ **Rapport PDF généré** : {pdf_path}")

                # Lire fichier pour download
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()

                st.download_button(
                    "📥 Télécharger PDF",
                    pdf_bytes,
                    "rapport_5g.pdf",
                    "application/pdf"
                )

            elif format_choice == "Excel":
                excel_path = generator.create_excel_report(report_data)
                st.success(f"✅ **Rapport Excel généré** : {excel_path}")

                # Lire fichier pour download
                with open(excel_path, 'rb') as f:
                    excel_bytes = f.read()

                st.download_button(
                    "📥 Télécharger Excel",
                    excel_bytes,
                    "rapport_5g.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"❌ Erreur génération rapport : {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ====== TÉLÉCHARGEMENTS DIRECTS ======
st.markdown("---")
st.subheader("💾 **Téléchargements Directs**")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Benchmark Modèles**")
    if 'model_results' in st.session_state and st.session_state.model_results is not None:
        csv = st.session_state.model_results.to_csv(index=False)
        st.download_button(
            "📥 Télécharger CSV",
            csv,
            "benchmark_modeles.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("Aucun résultat disponible")

with col2:
    st.markdown("**📈 Données Traitées**")
    if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
        # Limiter à 1000 lignes pour éviter fichiers trop gros
        csv_data = st.session_state.processed_data.tail(1000).to_csv(index=False)
        st.download_button(
            "📥 Télécharger CSV",
            csv_data,
            "donnees_traitees.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("Aucune donnée disponible")

with col3:
    st.markdown("**📋 Rapport HTML**")
    if st.session_state.get('full_report'):
        st.download_button(
            "📥 Télécharger HTML",
            st.session_state.full_report,
            "rapport_5g_complet.html",
            "text/html",
            use_container_width=True
        )
    else:
        st.info("Générer d'abord le rapport")

# ====== EXPORT CONFIGURATION ======
st.markdown("---")
st.subheader("⚙️ **Export Configuration Complète**")

if st.button("📦 **Exporter Configuration Projet**"):
    import json

    config = {
        'project': '5G Traffic Predictor PRO',
        'version': '2.0.0',
        'date': str(pd.Timestamp.now()),
        'data': {
            'sequence_length': st.session_state.get('sequence_length', 60),
            'prediction_horizon': st.session_state.get('prediction_horizon', 10),
            'targets': st.session_state.get('targets', []),
            'features': st.session_state.get('features', [])
        },
        'models': {
            'trained': len(st.session_state.get('trained_models', {})),
            'best': collect_session_data().get('best_model', 'N/A')
        }
    }

    st.download_button(
        "📥 Télécharger Config (JSON)",
        json.dumps(config, indent=2),
        "config_projet.json",
        "application/json"
    )

# ====== FORMATS DISPONIBLES ======
st.markdown("---")
st.info("""
**📁 Formats disponibles** :
- **HTML** : Rapport interactif avec graphiques
- **PDF** : Rapport professionnel imprimable
- **Excel** : Données brutes avec plusieurs onglets
- **CSV** : Export simple des résultats
- **JSON** : Configuration technique du projet
""")

# ====== DOCUMENTATION ======
with st.expander("📘 **Documentation Export**"):
    st.markdown("""
    ### Structure Rapports

    **HTML** :
    - Résumé exécutif
    - Tableau performance modèles
    - Recommandations 5G
    - Graphiques interactifs

    **PDF** :
    - Header professionnel
    - Métriques clés
    - Tableaux formatés
    - Footer avec métadonnées

    **Excel** :
    - Onglet "Résumé"
    - Onglet "Performance Modèles"
    - Onglet "Prédictions"
    - Onglet "Actions"

    ### Utilisation
    1. Sélectionner format souhaité
    2. Activer options (détails, graphiques)
    3. Générer rapport
    4. Télécharger ou prévisualiser
    """)

# ====== NAVIGATION ======
st.markdown("---")
col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
with col_nav2:
    if st.button("🏠 **Retour Accueil**", type="secondary", use_container_width=True):
        st.switch_page("app.py")
