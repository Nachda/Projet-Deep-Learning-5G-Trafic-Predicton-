# 🚀 Prédiction de trafic 5G 

 Deep Learning: Prédiction du trafic 5G – Projet GSTR2 (BAC+4) ENSA Tétouan 2025‑2026.

---
>
Dashboard professionnel Streamlit pour la **prédiction** Deep Learning du trafic réseau 5G (throughput, packets).

## 🎯 Contexte du projet

Ce projet a été réalisé dans le cadre du module **Deep Learning** à l’ENSA Tétouan (GSTR2, 2024‑2025).  
L’objectif est de construire un **pipeline complet** (prétraitement, entraînement, évaluation, déploiement Streamlit) pour la **prédiction du trafic 5G** (débit, nombre de paquets, saturation) à partir de fichiers MS Teams ou tout autre fichier.

---
## 📓 Notebook du rapport

Le notebook d’analyse détaillée ainsi que le rapport du projet sont disponible dans le dépôt :

- `Projet_DL_Nachda_Nourouddine.ipynb` 
- `Rapport de projet deep learning.pdf `
Ils contiennent :
  - exploration du dataset MS Teams,
  - tests de différents modèles (baselines, ML, Deep Learning),
  - interprétation des résultats et choix de l’architecture finale.

Vous pouvez l’ouvrir dans Jupyter / VS Code pour voir tout le raisonnement mathématique et expérimental derrière le dashboard Streamlit.


## 📂 Données & source Kaggle

Le projet utilise des captures réseau MS Teams issues d’un dataset externe.

### 1) Dataset complet (non inclus dans le repo)

Le dataset principal `MS_Teams_1.csv` n’est **pas inclus** dans ce dépôt (taille > 1 Go, limite GitHub).  

Pour utiliser la **version démo complète** de l’application :

1. Récupérer le fichier `MS_Teams_1.csv` depuis la source suivante.  
2. Le placer à la **racine du projet**, au même niveau que `app.py`.

### 2) Lien Kaggle

Le dataset complet provient de :

- **Kaggle – 5G Traffic Datasets** :  
  https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets  

Types de trafic disponibles :

- Streaming de jeux / jeux en ligne / métavers : Roblox, Zepeto, Battleground, Teamfight_Tactics, etc.  
- Streaming stocké : Amazon Prime, Netflix (`Netflix_1.csv`), YouTube.  
- Diffusion en direct / vidéoconférence : Google Meet, MS Teams (`MS_Teams_1.csv`, `MS_Teams_2.csv`), Zoom (`Zoom_1.csv`, `Zoom_2.csv`, `Zoom_3.csv`).

### 3) Utilisation du dataset MS Teams dans ce projet

Pour reproduire la démo :

1. Télécharger le dataset depuis Kaggle :  
   https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets  
2. Sauvegarder le fichier sous le nom **`MS_Teams_1.csv`**.  
3. Le placer à la **racine du projet**, au même niveau que `app.py` :

   ```text
   Projet-Deep-Learning-5G-Trafic-Predicton-/
   ├── app.py
   ├── MS_Teams_1.csv      ← ici
   ├── pages/
   ├── utils/
   └── ...

4. Lancer l’application et utiliser la page 1 pour charger ce fichier (ou un chargement automatique si ce comportement est implémenté dans le code).

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
git clone https://github.com/Nachda/Projet-Deep-Learning-5G-Trafic-Predicton-.git
cd Projet-Deep-Learning-5G-Trafic-Predicton-

"git clone ...":
Télécharge une copie complète de ton projet GitHub sur la machine locale (code, dossier, historique Git).
​
Après cette commande, un dossier Projet-Deep-Learning-5G-Trafic-Predicton- est créé dans le répertoire courant.
​
"cd Projet-Deep-Learning-5G-Trafic-Predicton-":

Se place à l’intérieur du dossier du projet dans le terminal.
Toutes les commandes suivantes (python, pip, streamlit) supposent que tu es dans ce dossier, là où se trouvent app.py, requirements.txt, etc.

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

a) Création
"python -m venv venv":

Demande à Python d’exécuter le module venv qui sert à créer des environnements virtuels.

Crée un dossier venv/ dans ton projet, qui contient :

une copie isolée de Python,son propre pip,
les dossiers où seront installées les bibliothèques de ce projet seulement.

But : éviter les conflits de versions entre différents projets (par exemple, TensorFlow 2.10 ici, 2.16 dans un autre projet)

b) Activation (Linux / macOS)
source venv/bin/activate

Modifie ton environnement de terminal pour utiliser le Python et le pip de venv au lieu de ceux du système.​

Le prompt change souvent en quelque chose comme (venv) user@pc:~/Projet-Deep-Learning-5G... pour montrer que l’environnement est actif.

c) Activation (Windows)
venv\Scripts\activate

Même idée que ci‑dessus, mais avec le chemin Windows.
​
Tu peux aussi utiliser .\venv\Scripts\activate dans PowerShell.
​
Pour sortir de l’environnement virtuel, on tape simplement : deactivate
Cette commande remet le terminal sur le Python système.


# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py

Une fois la commande exécutée :

un serveur local démarre sur http://localhost:8501 par défaut,

le terminal affiche l’URL et quelques logs,

le navigateur s’ouvre automatiquement (ou tu peux coller l’URL manuellement).
​

L’application reste active tant que le terminal reste ouvert (ou tant qu'on ne fais pas Ctrl+C pour arrêter le serveur).
