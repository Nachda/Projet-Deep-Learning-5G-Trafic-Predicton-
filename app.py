import streamlit as st
from pathlib import Path
import sys

# Config
st.set_page_config(
    page_title="5G Traffic Predictor PRO",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS avancés 
st.markdown("""
<style>
    /* Sidebar professionnel */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Cards métriques */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Boutons stylisés */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Section workflow */
    .workflow-step {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #e5e7eb;
        transition: all 0.3s;
    }
    
    .workflow-step:hover {
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .step-number {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        text-align: center;
        line-height: 40px;
        font-weight: bold;
        margin-right: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Header principal 
st.markdown("""
<div class="main-header">
    <h1>📡 5G Traffic Predictor PRO</h1>
    <p>Prédiction & Optimisation Réseau 5G par Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar 
with st.sidebar:
    st.markdown("### 📚 **Publications Scientifiques**")
    
    st.markdown("""
    **[Deep Learning for 5G Traffic Prediction]**  
    [IEEE Paper](https://ieeexplore.ieee.org/document/10134497)
    """)
    
    st.markdown("""
    **[LSTM Traffic Forecasting 5G]**  
    [IEEE](https://ieeexplore.ieee.org/document/10763791)
    """)
    
    st.markdown("""
    **[Transformer 5G Networks]**  
    [5G_Transformer](https://5g-ppp.eu/5g-transformer/)
    """)
    
    st.markdown("""
    **[Streamlit ML Apps]**  
    [Streamlit](https://streamlit.io/)
    """)

# Section Présentation 
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>📊 15+ Modèles</h3>
        <p>Baselines, ML, Deep Learning, Ensembles</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>🤖 Multi-Output</h3>
        <p>Prédiction simultanée de plusieurs métriques</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>⚡ Temps Réel</h3>
        <p>Prédictions et actions réseau automatisées</p>
    </div>
    """, unsafe_allow_html=True)

# Section Workflow 
st.markdown("---")
st.subheader("🎯 **Workflow Complet**")

workflow_steps = [
    ("📤 Upload Données", "Fichiers CSV Wireshark ou métriques pré-agrégées", "Page 1"),
    ("🔧 Preprocessing", "Resampling 1s, création features, normalisation", "Page 1"),
    ("🧠 Entraînement", "15+ modèles benchmark automatique", "Page 2"),
    ("🤖 Prédictions", "Multi-horizon avec intervalles de confiance", "Page 3"),
    ("📈 Analyse", "Comparaison modèles, feature importance", "Page 4"),
    ("⚡ Actions", "Recommandations réseau automatisées", "Page 5"),
    ("📋 Rapports", "Export PDF/Excel/HTML professionnels", "Page 6")
]

for i, (title, desc, page) in enumerate(workflow_steps, 1):
    st.markdown(f"""
    <div class="workflow-step">
        <span class="step-number">{i}</span>
        <strong>{title}</strong> - {desc}
        <br><small style="color: #666;">→ {page}</small>
    </div>
    """, unsafe_allow_html=True)

# Section Fonctionnalités Clés 
st.markdown("---")
st.subheader("✨ **Fonctionnalités Avancées**")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 🎯 Analyse & Prédiction
    - **Détection automatique** format de données
    - **Séquences temporelles** configurables (30-120s)
    - **Prédictions multi-horizon** (1-60s)
    - **Intervalles de confiance** pour quantifier l'incertitude
    - **Détection d'anomalies** en temps réel
    """)

with col2:
    st.markdown("""
    #### ⚡ Optimisation Réseau
    - **Recommandations QoS** dynamiques
    - **Gestion bande passante** prédictive
    - **Alertes automatiques** sur seuils
    - **Simulation impact** des actions
    - **Export rapports** professionnels (PDF/Excel/HTML)
    """)

# Section Démarrage Rapide 
st.markdown("---")
st.subheader("🚀 **Démarrage Rapide**")

st.markdown("""
1. **📤 Page 1** : Upload ton fichier CSV OU utilise le dataset démo MS Teams
2. **🧠 Page 2** : Entraîne 15+ modèles en 1 clic (benchmark automatique)
3. **🤖 Page 3** : Visualise prédictions multi-horizon en temps réel
4. **📊 Page 4** : Compare les performances + feature importance
5. **⚡ Page 5** : Applique recommandations réseau automatisées
6. **📋 Page 6** : Exporte rapports professionnels (PDF/Excel/HTML)
""")

# Bouton CTA 
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🎯 **Commencer l'Analyse** →", type="primary", use_container_width=True):
        st.switch_page("pages/1_📊_Data_Upload_Preprocessing.py")

# Footer avec LIENS RÉELS
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📚 Documentation**
    - [📖 Guide Utilisateur](https://docs.streamlit.io/)
    """)

with col2:
    st.markdown("""
    **🔗 Ressources**
    - [🐙 GitHub Repository](https://github.com/)  
    - [📊 Dataset 5G Kaggle](https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets)
    """)

st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
    <p style="margin: 0; color: #666;">
        © 2024 5G Traffic Predictor PRO | Powered by TensorFlow 2.13 + Streamlit 1.28
    </p>
</div>
""", unsafe_allow_html=True)
