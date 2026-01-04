import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import gc
import traceback
import io
from sklearn.preprocessing import RobustScaler
from styles import inject_global_styles, page_header  

inject_global_styles()  
page_header("📊 Upload & Preprocessing", "Étape 1/6")  

# =========================
# PATHS & IMPORTS
# =========================
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'utils'))
sys.path.insert(0, str(BASE_DIR))

try:
    from utils import DataProcessor
    st.success("✅ DataProcessor OK")
except Exception:
    st.error("❌ utils/DataProcessor.py manquant")
    st.stop()

# =========================
# ÉTAT GLOBAL
# =========================
default_keys = {
    "df_loaded": False,
    "total_lines": 0,
    "processed_data": None,
    "raw_metrics": None,
    "df": None,
    "targets": None,
    "features": None,
    "prediction_mode": "multi",
    "source_type": None,
}

for k, v in default_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# 🎯 OPTION A : DATASET DÉMO OU UPLOAD
# =========================
st.subheader("📁 **Source des Données**")
data_source = st.radio(
    "Choisir la source",
    options=["📂 Upload mon fichier CSV", "🎓 Utiliser dataset démo (MS Teams 5G)"],
    index=1
)

# RESET si l'utilisateur change de source
if 'last_data_source' not in st.session_state:
    st.session_state.last_data_source = data_source
elif st.session_state.last_data_source != data_source:
    st.session_state.df_loaded = False
    st.session_state.df = None
    st.session_state.last_data_source = data_source

if data_source == "🎓 Utiliser dataset démo (MS Teams 5G)":
    demo_files = list(BASE_DIR.glob("*.csv"))
    if not demo_files:
        st.error("❌ Aucun fichier CSV démo trouvé à la racine !")
        st.info("💡 Place un fichier CSV (ex: MS_Teams.csv) dans le dossier racine.")
        st.stop()
    
    file_to_load = demo_files[0]
    st.success(f"✅ **Dataset démo** : {file_to_load.name} ({file_to_load.stat().st_size/1e6:.1f} MB)")
    st.session_state.source_type = "demo"
    st.session_state.file_to_load = file_to_load

else:
    uploaded_file = st.file_uploader(
        "📤 Upload ton fichier CSV 5G",
        type=["csv"],
        help="Colonnes attendues : Time, Length (minimum). Max 2GB."
    )
    
    if uploaded_file is None:
        st.warning("⚠️ Upload un fichier pour continuer")
        st.stop()
    
    st.session_state.uploaded_file_data = uploaded_file.getvalue()
    st.session_state.uploaded_file_name = uploaded_file.name
    st.session_state.source_type = "upload"
    st.success(f"✅ **Fichier uploadé** : {uploaded_file.name} ({len(st.session_state.uploaded_file_data)/1e6:.1f} MB)")

# =========================
# BOUTON CLEAR
# =========================
if st.button("🗑️ Clear Session", type="secondary"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# =========================
# ÉTAPE 1 : CHARGEMENT CSV
# =========================
if not st.session_state.df_loaded:
    if st.session_state.source_type == "demo":
        button_text = f"🚀 **ÉTAPE 1 : Charger {st.session_state.file_to_load.name}**"
    else:
        button_text = f"🚀 **ÉTAPE 1 : Charger {st.session_state.uploaded_file_name}**"
    
    if st.button(button_text, type="primary"):
        with st.spinner("🔄 Chargement par chunks..."):
            try:
                chunks = []
                total_lines = 0
                
                if st.session_state.source_type == "demo":
                    for chunk in pd.read_csv(st.session_state.file_to_load, chunksize=50000, low_memory=False):
                        if 'No.' in chunk.columns:
                            chunk['No.'] = pd.to_numeric(chunk['No.'], errors='coerce').astype('int32')
                        if 'Length' in chunk.columns:
                            chunk['Length'] = pd.to_numeric(chunk['Length'], errors='coerce').astype('float32')
                        chunks.append(chunk)
                        total_lines += len(chunk)
                        gc.collect()
                else:
                    file_like_object = io.BytesIO(st.session_state.uploaded_file_data)
                    for chunk in pd.read_csv(file_like_object, chunksize=50000, low_memory=False):
                        if 'No.' in chunk.columns:
                            chunk['No.'] = pd.to_numeric(chunk['No.'], errors='coerce').astype('int32')
                        if 'Length' in chunk.columns:
                            chunk['Length'] = pd.to_numeric(chunk['Length'], errors='coerce').astype('float32')
                        chunks.append(chunk)
                        total_lines += len(chunk)
                        gc.collect()
                
                df = pd.concat(chunks, ignore_index=True)
                del chunks
                gc.collect()

                st.session_state.df = df
                st.session_state.df_loaded = True
                st.session_state.total_lines = total_lines

                st.success(f"✅ **{total_lines:,} lignes chargées**")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Erreur chargement : {e}")
                st.code(traceback.format_exc())
                st.stop()

# =========================
# DIAGNOSTIC : VÉRIFICATION DES DONNÉES BRUTES
# =========================
if st.session_state.df_loaded:
    df = st.session_state.df
    
    st.info(f"📊 **DataFrame prêt** : {len(df):,} lignes × {len(df.columns)} colonnes")

    time_cols = [col for col in df.columns if 'time' in col.lower()]
    length_cols = [col for col in df.columns if 'length' in col.lower()]

    st.subheader("👁️ **Aperçu Données Brutes**")
    st.dataframe(df.head(5), use_container_width=True)

    # =========================
    # DIAGNOSTIC DES DONNÉES D'ENTRÉE 
    # =========================
    st.subheader("🔍 **Diagnostic des Données d'Entrée**")

    if 'Length' in df.columns:
        st.write("**Colonne Length (brute) :**")
        length_stats = df['Length'].describe()
        st.write(length_stats)
    
        # ANALYSE INTELLIGENTE BASÉE SUR LES VRAIES STATS
        length_min, length_max = df['Length'].min(), df['Length'].max()
        length_median = df['Length'].median()
    
        st.info(f"""
        **📊 Analyse de la colonne Length :**
        • **Min :** {length_stats['min']:.0f} bytes
        • **Max :** {length_stats['max']:.0f} bytes  
        • **Médiane :** {length_stats['50%']:.0f} bytes
        • **Moyenne :** {length_stats['mean']:.1f} bytes
    
        **✅ Interprétation CORRECTE :**
        """)
    
        # LOGIQUE D'ANALYSE 
        analysis_points = []
    
        if length_min >= 0 and length_max <= 1500:
            analysis_points.append("✅ **Tailles de paquets réseaux normales** (Ethernet MTU ≈ 1500 bytes)")
    
        if length_min < 60:
            analysis_points.append("⚠️ **Certains paquets très petits** (< 60 bytes) - peut être du trafic de contrôle")
    
        if length_max > 9000:
            analysis_points.append("⚠️ **Paquets jumbo frames détectés** (> 9000 bytes) - vérifier la configuration réseau")
    
        if length_min >= 0 and length_max <= 1 and length_max > 0.1:
            analysis_points.append("❌ **PROBLÈME : Données déjà normalisées** (valeurs 0-1)")
    
        if length_min < 0:
            analysis_points.append("❌ **ERREUR : Valeurs négatives** - données corrompues")
    
        # Afficher tous les points d'analyse
        for point in analysis_points:
            st.write(point)
    
        # CONCLUSION
        if "✅ **Tailles de paquets réseaux normales**" in analysis_points:
            st.success("""
            **🎯 CONCLUSION : Données PARFAITES pour les métriques 5G !**
            Vos données `Length` sont en **bytes réels** → vous pouvez activer 
            **"Créer métriques 5G dérivées"** sans problème.
            """)
        elif "❌ **PROBLÈME : Données déjà normalisées**" in analysis_points:
            st.error("""
            **🚨 PROBLÈME DÉTECTÉ :** 
            Vos données `Length` sont déjà normalisées (valeurs entre 0 et 1).
            **Action recommandée :** Désactivez "Créer métriques 5G dérivées".
            """)
    
        # Aperçu des valeurs
        st.write(f"**🔍 5 premières valeurs Length :** {df['Length'].head().tolist()}")

    # =========================
    # CHOIX DE LA STRATÉGIE
    # =========================
    st.markdown("---")
    st.subheader("⚙️ **Stratégie de Préparation**")
    
    col_strat1, col_strat2 = st.columns(2)
    
    with col_strat1:
        use_metrics = st.checkbox(
            "Créer métriques 5G dérivées",
            value=True,
            help="À DÉSACTIVER si Length est déjà normalisée (0-1)"
        )
    
    with col_strat2:
        force_raw = st.checkbox(
            "Utiliser données brutes (sans normalisation)",
            value=False,
            help="Pour garder les valeurs originales"
        )
    
    if not time_cols or not length_cols:
        st.warning("⚠️ Colonnes Time/Length manquantes → Métriques 5G indisponibles")
        use_metrics = False

    # =========================
    # BOUTON PRÉPARATION
    # =========================
    if st.button("🎯 **ÉTAPE 2 : Préparer Dataset ML**", type="primary", key="prepare_ml"):
        try:
            with st.spinner("🔄 Création features + normalisation..."):
                gc.collect()

                # --- CAS 1 : MÉTRIQUES 5G ---
                if use_metrics and time_cols and length_cols:
                    # 1. Extraire les colonnes utiles
                    useful_cols = [time_cols[0], length_cols[0]]
                    if 'Protocol' in df.columns:
                        useful_cols.append('Protocol')
                    
                    df_light = df[useful_cols].copy()
                    
                    # 2. VÉRIFIER que Length n'est pas normalisée
                    length_values = df_light[length_cols[0]]
                    if length_values.between(0, 1).all() and length_values.max() > 0.1:
                        st.error("""
                        ❌ **ERREUR : Length est déjà normalisée (0-1)**
                        
                        Les métriques calculées seront incorrectes.
                        **Action recommandée :**
                        1. Décochez "Créer métriques 5G dérivées"
                        2. Utilisez directement les colonnes numériques existantes
                        3. OU vérifiez votre fichier CSV source
                        """)
                        st.stop()
                    
                    processor = DataProcessor(df_light)
                    
                    # 3. Calculer les métriques RÉELLES
                    st.info("📈 Création métriques 5G (freq='1S')...")
                    processor.create_traffic_metrics(
                        time_column=time_cols[0],
                        length_column=length_cols[0],
                        freq='1S'
                    )
                    
                    # 4. SAUVEGARDER les données BRUTES
                    raw_metrics_df = processor.get_processed_data()
                    st.session_state.raw_metrics = raw_metrics_df.copy()
                    
                    st.write("**✅ Métriques brutes calculées :**")
                    st.dataframe(raw_metrics_df.head(), use_container_width=True)
                    st.write(f"**Colonnes créées :** {raw_metrics_df.columns.tolist()}")
                    
                    # 5. NORMALISATION (sauf si désactivée)
                    if not force_raw:
                        st.info("✅ Normalisation (RobustScaler)...")
                        processor.normalize_data('robust')
                        normalized_df = processor.get_processed_data()
                        
                        st.write("**📊 Après normalisation :**")
                        st.dataframe(normalized_df.head(), use_container_width=True)
                        
                        st.session_state.processed_data = normalized_df
                    else:
                        st.info("⏭️ Normalisation désactivée")
                        st.session_state.processed_data = raw_metrics_df
                    
                    final_df = st.session_state.processed_data

                # --- CAS 2 : DONNÉES BRUTES (sans métriques) ---
                else:
                    st.info("ℹ️ Utilisation colonnes numériques brutes")
                    
                    numeric_df = df.select_dtypes(include='number').copy()
                    numeric_df = numeric_df.fillna(method='ffill').fillna(0)
                    
                    st.write("**📊 Données numériques brutes :**")
                    st.dataframe(numeric_df.head(), use_container_width=True)
                    
                    st.session_state.raw_metrics = numeric_df.copy()
                    
                    if not force_raw:
                        st.info("✅ Normalisation (RobustScaler)...")
                        scaler = RobustScaler()
                        numeric_normalized = scaler.fit_transform(numeric_df)
                        numeric_normalized_df = pd.DataFrame(
                            numeric_normalized, 
                            columns=numeric_df.columns,
                            index=numeric_df.index
                        )
                        st.session_state.processed_data = numeric_normalized_df
                        
                        st.write("**🎯 Après normalisation :**")
                        st.dataframe(numeric_normalized_df.head(), use_container_width=True)
                    else:
                        st.session_state.processed_data = numeric_df
                    
                    final_df = st.session_state.processed_data
                
                # --- SÉLECTION DES CIBLES (commun aux deux cas) ---
                st.subheader("🎯 **Choix des Cibles pour la Prédiction**")
                
                final_numeric_cols = final_df.select_dtypes(include='number').columns.tolist()
                st.write(f"**Colonnes numériques disponibles ({len(final_numeric_cols)}) :**")
                st.json(final_numeric_cols)
                
                # Déterminer les cibles par défaut intelligemment
                default_targets = []
                if "packet_count" in final_numeric_cols:
                    default_targets.append("packet_count")
                if "throughput_mbps" in final_numeric_cols:
                    default_targets.append("throughput_mbps")
                if len(default_targets) == 0 and len(final_numeric_cols) >= 2:
                    default_targets = final_numeric_cols[:2]
                
                selected_targets = st.multiselect(
                    "Sélectionnez les colonnes à prédire (cibles) :",
                    options=final_numeric_cols,
                    default=default_targets,
                    help="Choisissez au moins une cible. Multi-output possible.",
                    key="targets_selector_final"
                )
                
                if not selected_targets:
                    st.warning("⚠️ Veuillez sélectionner au moins une cible")
                    st.stop()
                
                # Calculer les features automatiquement
                feature_cols = [c for c in final_numeric_cols if c not in selected_targets]
                
                # Sauvegarder dans l'état global
                st.session_state.targets = selected_targets
                st.session_state.features = feature_cols
                
                st.success(
                    f"🎉 **Configuration terminée !**\n"
                    f"• {len(final_df):,} échantillons\n"
                    f"• {len(feature_cols)} features → {len(selected_targets)} cibles"
                )
                
                st.info(f"**🎯 Cibles :** {selected_targets}")
                st.info(f"**🔧 Features :** {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}")
                
                # Afficher un aperçu du dataset final
                st.subheader("📋 **Aperçu du Dataset Final**")
                st.dataframe(final_df.head(), use_container_width=True)
                
                # Message de navigation
                st.success("✅ **Dataset prêt ! Page 2 → Entraîner modèles**")
                
                if st.button("➡️ Aller à la Page 2 - Entraînement", type="secondary"):
                    st.switch_page("pages/2_🧠_Model_Training.py")

        except Exception as e:
            st.error(f"❌ Erreur préparation : {str(e)}")
            st.code(traceback.format_exc())

# =========================
# COMPARAISON : BRUTES vs NORMALISÉES (si disponible)
# =========================
if st.session_state.get('raw_metrics') is not None and st.session_state.get('processed_data') is not None:
    st.markdown("---")
    st.subheader("🔍 **Comparaison : Données Brutes vs Normalisées**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Données Brutes (Valeurs Physiques)**")
        st.dataframe(st.session_state.raw_metrics.head(), use_container_width=True)
        st.caption("Unités réelles : packets/s, Mbps, bytes, etc.")
    
    with col2:
        st.markdown("**🎯 Données Normalisées (Pour ML)**")
        st.dataframe(st.session_state.processed_data.head(), use_container_width=True)
        st.caption("Centrées-réduites (RobustScaler) - meilleure convergence")
    
    # Exemple de conversion pour une métrique courante
    common_cols = set(st.session_state.raw_metrics.columns) & set(st.session_state.processed_data.columns)
    if common_cols:
        sample_col = list(common_cols)[0]
        if sample_col in st.session_state.raw_metrics.columns and sample_col in st.session_state.processed_data.columns:
            raw_sample = st.session_state.raw_metrics[sample_col].iloc[0]
            norm_sample = st.session_state.processed_data[sample_col].iloc[0]
            
            st.info(f"""
            **Exemple de transformation pour '{sample_col}' :**
            - **Valeur brute** : `{raw_sample:.6f}` (unité physique)
            - **Valeur normalisée** : `{norm_sample:.6f}` (sans unité, échelle standard)
            """)

# =========================
# AFFICHAGE FINAL (si déjà préparé)
# =========================
if st.session_state.get('processed_data') is not None:
    st.markdown("---")
    st.subheader("📈 **Résumé Dataset ML (Déjà préparé)**")

    processed_df = st.session_state.processed_data

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 Échantillons", f"{len(processed_df):,}")
        st.metric("📏 Colonnes totales", len(processed_df.columns))

    with col2:
        if 'timestamp' in processed_df.columns or processed_df.index.name == 'Time':
            st.metric("⏱️ Fréquence", "1 seconde")
        if st.session_state.get('targets'):
            st.metric("🎯 Cibles", len(st.session_state.targets))
    
    if st.session_state.get('targets'):
        st.info(f"**🎯 Cibles configurées :** {st.session_state.targets}")
        if st.session_state.get('features'):
            st.info(f"**🔧 Nombre de features :** {len(st.session_state.features)}")
    
    st.success("✅ **Dataset déjà préparé. Vous pouvez passer à la page 2.**")