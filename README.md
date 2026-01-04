# 🚀 5G Traffic Predictor Pro

Dashboard professionnel Streamlit pour la **prédiction** et l’**optimisation** du trafic réseau 5G (throughput, packets) avec Machine Learning et Deep Learning.

---

## ✨ Fonctionnalités

### 📊 Exploration & Preprocessing

- Upload de fichiers **CSV / Excel**
- Détection automatique du format de données (brut `Time, Length` ou pré‑agrégé)
- Prétraitement automatique :
  - Resampling temporel (1 seconde)
  - Agrégation réseau (throughput, packet_count, stats de paquet)
  - Normalisation / scaling des variables cibles
- Visualisations interactives pour l’EDA (distributions, séries temporelles)

### 🧠 Entraînement Multi‑Modèles

- Baselines :
  - Persistence
  - Moving Average
- Modèles ML :
  - Linear Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Deep Learning (TensorFlow / Keras) :
  - MLP
  - LSTM
  - GRU
  - BiLSTM
  - CNN‑LSTM
  - Transformer‑like
- Ensembles :
  - Voting
  - Stacking
- Entraînement avec:
  - séquences glissantes (lookback configurable)
  - horizon de prédiction multi‑pas (ex. 10 s)
  - early stopping pour les modèles DL
- Comparaison automatique (MAE, RMSE, R², temps d’entraînement)

### 🤖 Prédictions Temps Réel

- Prédictions **multi‑horizon** (ex. 1–10 secondes) sur les dernières fenêtres de test
- Prédictions **multi‑cibles** (ex. `packet_count`, `throughput_mbps`)
- Vue **normalisée** (debug modèle) et vue **brute** (base décision opérateur)
- Graphiques interactifs Plotly (réel vs prédit)
- Indicateurs simples (MAE, statut réseau)

### ⚡ Actions Réseau 5G

- Page dédiée aux **actions opérateur** :
  - Santé réseau (health score) à partir des métriques prédites
  - Détection de saturation / dégradation
  - Boutons d’actions simulées (QoS, slicing, buffers, priorisation trafic)
- Vue synthétique du **meilleur modèle** (selon score composite)

### 📋 Rapports Professionnels

- Génération automatique :
  - **PDF** (ReportLab)
  - **Excel** multi‑onglets
  - **HTML** interactif
  - **JSON** (config projet / actions)
- Contenu des rapports :
  - Tableau de performance des modèles (MAE, R², `Train_Time_s`)
  - Meilleur modèle + métriques moyennes
  - Score de santé réseau
  - Recommandations 5G

---

## 🚀 Installation Rapide

### Prérequis

- Python **3.9+**
- `pip`
- 8 Go RAM recommandés

### Installation

```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/5g-traffic-predictor.git
cd 5g-traffic-predictor

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
