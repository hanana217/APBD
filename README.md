# SADOP - Système Autonome de Diagnostic et d'Optimisation de Performance SQL

Système complet d'optimisation de bases de données MySQL utilisant **3 vrais modèles de Machine Learning** (Random Forest, XGBoost, Logistic Regression) et **Reinforcement Learning (DQN)**.

---

## 🚀 Démarrage rapide

```bash
cd docker
docker-compose up --build
```

Une fois tous les services démarrés :

| Service       | URL                          | Description                        |
|---------------|------------------------------|------------------------------------|
| **Frontend**  | http://localhost:8501         | Interface Streamlit                |
| **Backend**   | http://localhost:8000         | API FastAPI (+ docs Swagger)       |
| **API Docs**  | http://localhost:8000/docs    | Documentation interactive Swagger  |
| **Agent RL**  | http://localhost:8002         | API de l'agent RL                  |
| **Grafana**   | http://localhost:3000         | Monitoring (admin/admin)           |
| **Prometheus**| http://localhost:9090         | Métriques                          |
| **MySQL**     | localhost:3308                | Base de données (apbd_user/apbd_pass) |

---

## 📁 Structure du projet

```
APBD/
├── docker/                          # Infrastructure Docker
│   ├── docker-compose.yml           # ⭐ Point d'entrée unique
│   ├── mysql/
│   │   └── my.ini                   # Configuration MySQL optimisée
│   └── monitoring/
│       └── prometheus.yml           # Configuration Prometheus
│
├── apbd_interface/                  # Application principale
│   ├── backend/                     # API FastAPI
│   │   ├── Dockerfile
│   │   ├── main.py                  # API avec 3 vrais modèles ML
│   │   ├── generate_dataset.py      # Générateur de dataset d'entraînement
│   │   └── requirements.txt
│   └── frontend/                    # Interface Streamlit
│       ├── Dockerfile
│       ├── app.py                   # Dashboard complet (7 pages)
│       └── requirements.txt
│
├── agent/                           # Agent Reinforcement Learning
│   ├── Dockerfile
│   ├── agent_api.py                 # API Flask de l'agent
│   ├── config.py                    # Configuration (env vars)
│   ├── env_enhanced.py              # Environnement Gymnasium
│   ├── mysql_utils.py               # Utilitaires MySQL
│   ├── train_agent.py               # Script d'entraînement DQN
│   └── requirements.txt
│
└── sql/                             # Scripts SQL
    ├── schema.sql                   # Schéma complet BDD POS + données
    ├── queries_bad.sql              # Requêtes lentes (test)
    └── indexes_bad.sql              # Index sous-optimaux (test)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Frontend Streamlit                  │
│         (Dashboard + Chat IA + 7 pages)             │
│                  :8501                               │
└─────────────────┬───────────────────────────────────┘
                  │ HTTP
┌─────────────────▼───────────────────────────────────┐
│                Backend FastAPI                       │
│   3 Modèles ML : RF + XGBoost + LR                 │
│   Simulateur RL + Chat Agent + CSV Upload           │
│                  :8000                               │
└────────┬────────────────────────┬────────────────────┘
         │                        │
┌────────▼────────┐    ┌──────────▼──────────┐
│   MySQL 8.0     │    │   Agent RL (DQN)    │
│   BDD POS       │◄───│   Optimisation      │
│   :3306         │    │   d'index           │
│                 │    │   :8000 (→ :8002)   │
└─────────────────┘    └─────────────────────┘
```

---

## 🔧 Fonctionnalités

### 🔍 Prédiction ML (3 modèles entraînés)
- **Random Forest** 🌲 : Classifieur d'ensemble robuste
- **XGBoost** 🚀 : Gradient boosting haute performance
- **Logistic Regression** 📐 : Modèle linéaire interprétable
- Chaque modèle prédit si une requête SQL sera **lente** ou **rapide**
- Extraction automatique de **20 features** (JOINs, sous-requêtes, GROUP BY, taille des tables...)
- Comparaison des 3 modèles côte à côte avec métriques (accuracy, F1-score, AUC)

### 📂 Analyse par fichier CSV
- Upload de fichiers CSV contenant des requêtes SQL
- Prédiction batch avec les 3 modèles simultanément
- Résumé statistique des résultats

### 🤖 Optimisation RL (Reinforcement Learning)
- Agent DQN qui apprend à **créer/supprimer des index**
- 3 actions : CREATE, DROP, NOOP
- Récompenses basées sur l'amélioration réelle des performances
- Max 5 index simultanés

### 💬 Assistant IA
- Chat interactif pour analyser des requêtes
- Suggestions d'optimisation automatique
- Recommandations d'index

### 📊 Monitoring
- Dashboard temps réel avec métriques des modèles ML
- F1-Score comparatif des 3 modèles
- Historique des optimisations
- Graphiques d'évolution (Plotly)

---

## 🧠 Pipeline ML

1. **Génération du dataset** : `generate_dataset.py` crée un dataset synthétique de 3000 requêtes SQL avec 20 features
2. **Entraînement** : Au démarrage du backend, les 3 modèles sont entraînés sur le dataset (train/test split 80/20)
3. **Prédiction** : Pour chaque requête SQL, les features sont extraites puis passées aux 3 modèles
4. **Fallback** : Si `generate_dataset.py` échoue, un dataset minimal de 1000 lignes est généré automatiquement

---

## ⚙️ Configuration

Toutes les configurations se font via **variables d'environnement** (définies dans le `docker-compose.yml`) :

| Variable         | Défaut       | Description              |
|------------------|-------------|--------------------------|
| `MYSQL_HOST`     | `mysql`     | Hôte de la BDD           |
| `MYSQL_PORT`     | `3306`      | Port de la BDD           |
| `MYSQL_USER`     | `apbd_user` | Utilisateur MySQL        |
| `MYSQL_PASSWORD`  | `apbd_pass` | Mot de passe MySQL       |
| `MYSQL_DATABASE`  | `pos`       | Nom de la base           |
| `BACKEND_URL`    | `http://backend:8000` | URL du backend (frontend) |

---

## 🛑 Arrêt

```bash
cd docker
docker-compose down
```

Pour supprimer aussi les données :
```bash
docker-compose down -v
```

---

## 📦 Base de données POS

Tables principales : `admin`, `clients`, `wilayas`, `products`, `promotions`, `offers`, `cart`, `orders`, `claims`, `comments`, `rating`, `favorites`, `returns`, `inbox`, `query_logs`

---

**SADOP v5.0** | © APBD Team


