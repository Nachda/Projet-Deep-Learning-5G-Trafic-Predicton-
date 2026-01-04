# pages/5_⚡_Network_Actions.py 

import streamlit as st
import pandas as pd
import numpy as np

from styles import inject_global_styles, page_header
inject_global_styles()
page_header("⚡ Actions 5G Automatisées", "Étape 5/6 - Décisions Opérateur")

# SIDEBAR
with st.sidebar:
    st.markdown("### ⚡ **Statut Réseau**")
    if st.session_state.get('models_trained', False):
        results_df = st.session_state.model_results

        # Si Composite_Score dispo (page 4), l'utiliser, sinon MAE
        if 'Composite_Score' in results_df.columns:
            best = results_df.sort_values('Composite_Score', ascending=False).iloc[0]
        else:
            best = results_df.nsmallest(1, 'MAE').iloc[0]

        st.success(f"✅ **{best['Model'].split()[0]}** actif")
        st.metric("📉 MAE", f"{best['MAE']:.4f}")
        if 'Train_Time_s' in best:
            st.metric("⏱️ Temps train (s)", f"{best['Train_Time_s']:.1f}")

    st.markdown("---")
    st.button("📈 Page 4", key="page4")
    st.button("🏠 Accueil", key="home5")

# VÉRIFICATIONS
required_keys = ["model_results", "trained_models", "target_scaler",
                 "targets", "X_test", "y_test"]
if any(k not in st.session_state or st.session_state[k] is None for k in required_keys):
    st.error("❌ **Pages 2-4 requises**")
    st.stop()

# RÉCUPÉRATION
results_df = st.session_state.model_results
if 'Composite_Score' in results_df.columns:
    best_model = results_df.sort_values('Composite_Score', ascending=False).iloc[0]['Model']
else:
    best_model = results_df.nsmallest(1, 'MAE').iloc[0]['Model']

target_scaler = st.session_state.target_scaler
targets = st.session_state.targets
X_test, y_test = st.session_state.X_test, st.session_state.y_test

n_targets = len(targets)
st.success(f"🎯 **{n_targets} cibles** : {', '.join(targets)}")

# 🔥 1. ÉTAT RÉSEAU ACTUEL (basé sur DERNIÈRES données)
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white; padding: 1.5rem; border-radius: 12px;">
    <h2>📊 État Réseau Temps Réel</h2>
</div>
""", unsafe_allow_html=True)

# Dénormalisation MULTI-CIBLES sur les 10 derniers pas
n_last = min(10, y_test.shape[1])
y_recent_scaled = y_test[-1:, :n_last, :].reshape(-1, n_targets)
y_recent_real = target_scaler.inverse_transform(y_recent_scaled)
y_recent_real = y_recent_real.reshape(1, n_last, n_targets)[0]

cible1_mean = float(np.mean(y_recent_real[:, 0])) if n_targets > 0 else 0.0
cible2_mean = float(np.mean(y_recent_real[:, 1])) if n_targets > 1 else 0.0

# Health score simple (0-100) basé sur cible1 (ex: throughput)
# >50 très bon, entre 5 et 50 ok, <5 mauvais
if cible1_mean >= 50:
    health_score = 90.0
elif cible1_mean >= 5:
    health_score = 70.0
else:
    health_score = 40.0

col1, col2, col3 = st.columns(3)
col1.metric(f"📡 {targets[0] if n_targets>0 else 'Cible1'}", f"{cible1_mean:.1f}")
col2.metric(f"📦 {targets[1] if n_targets>1 else 'Cible2'}", f"{cible2_mean:.1f}")
col3.metric("🩺 Health Score", f"{health_score:.1f}/100")

# 🛠️ 2. RECOMMANDATIONS BASÉES SUR RÉSULTATS
st.markdown("""
<div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white; padding: 1.5rem; border-radius: 12px;">
    <h2>🎯 Recommandations Opérateur</h2>
</div>
""", unsafe_allow_html=True)

if cible1_mean > 50:
    st.error("🔴 **SATURATION** → Ajouter capacité RAN")
elif cible1_mean < 5:
    st.warning("🟡 **DÉGRADATION** → Diagnostic gNB")
else:
    st.success("🟢 **NOMINAL** → Surveillance")

st.info(f"""
**Modèle leader** : {best_model}  

**Actions selon trafic actuel** :
• {targets[0] if n_targets>0 else 'Cible1'} : {cible1_mean:.1f}  
• {targets[1] if n_targets>1 else 'Cible2'} : {cible2_mean:.1f}  
**Health Score estimé** : {health_score:.1f}/100
""")

# 🤖 3. ACTIONS INTERACTIVES
st.markdown("""
<div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
            color: white; padding: 1.5rem; border-radius: 12px;">
    <h2>⚡ Actions Automatisées</h2>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📱 UE", "🏺 Buffers", "🌐 Slicing", "⚙️ QoS"])

with tab1:
    st.subheader("📱 Optimisation UE")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 **Activer URLLC**", type="primary"):
            st.success("✅ URLLC activé → Latence <1ms")
    with col2:
        if st.button("📺 **Booster eMBB**", type="primary"):
            st.success("✅ eMBB boosté → Débit >100Mbps")

with tab2:
    st.subheader("🏺 Gestion Buffers")
    buffer_current = cible1_mean
    col1, col2, col3 = st.columns(3)
    col1.metric("Actuel", f"{buffer_current:.1f}")
    col2.metric("Recommandé", f"{buffer_current*1.2:.1f}", f"+20%")
    if col3.button("🔧 **Ajuster Buffers**", type="primary"):
        st.balloons()
        st.success("✅ Buffers ajustés +20%")

with tab3:
    st.subheader("🌐 Network Slicing")
    col1, col2, col3 = st.columns(3)
    if col1.button("🎮 **URLLC Slice**", type="primary"):
        st.success("✅ Slice Gaming/Autonomous activé")
    if col2.button("📱 **eMBB Slice**", type="primary"):
        st.success("✅ Slice Streaming/VR activé")
    if col3.button("🏠 **mMTC Slice**", type="primary"):
        st.success("✅ Slice IoT/SmartCity activé")

with tab4:
    st.subheader("⚙️ QoS Dynamique")
    if st.button("🎯 **Prioriser Critique**", type="primary", use_container_width=True):
        st.success(f"✅ **{best_model}** → Trafic critique priorisé")

# 💾 EXPORT
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    config = {
        "best_model": best_model,
        f"{targets[0] if n_targets>0 else 'cible1'}": cible1_mean,
        f"{targets[1] if n_targets>1 else 'cible2'}": cible2_mean,
        "status": "NOMINAL" if cible1_mean > 5 else "DÉGRADATION",
        "health_score": health_score,
        "actions": "AUTO"
    }
    st.download_button(
        "📥 **Config 5G**",
        pd.DataFrame([config]).to_json(indent=2),
        "5g_actions.json", "application/json"
    )

with col2:
    st.button("📋 **Page 6 : Rapports** →", type="primary")
