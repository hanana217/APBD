# backend/main.py - API SADOP avec vrais modèles ML (RF, XGBoost, LR)
# ==================================================================
# Remplace la simulation par de vrais modèles entraînés sur le dataset POS.
# Conserve les endpoints RL (simulation) pour compatibilité frontend.

import os
import re
import math
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import mysql.connector
from datetime import datetime
import time
import io

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGBOOST_AVAILABLE = False

# ==================== CONFIGURATION ====================

MYSQL_CONFIG = {
    'host': os.environ.get('MYSQL_HOST', 'localhost'),
    'port': int(os.environ.get('MYSQL_PORT', '3306')),
    'user': os.environ.get('MYSQL_USER', 'apbd_user'),
    'password': os.environ.get('MYSQL_PASSWORD', 'apbd_pass'),
    'database': os.environ.get('MYSQL_DATABASE', 'pos'),
    'autocommit': True
}

SLOW_QUERY_THRESHOLD = 0.5

RL_CONFIG = {
    'max_indexes': 5,
    'creation_cost': 0.02,
    'drop_penalty': 0.01,
    'episode_length': 25
}

# ============================================================
# Metadata POS
# ============================================================

POS_TABLE_INFO = {
    'admin':           {'rows': 5,     'indexes': 1, 'has_text': False},
    'clients':         {'rows': 1000,  'indexes': 4, 'has_text': False},
    'wilayas':         {'rows': 58,    'indexes': 1, 'has_text': False},
    'products':        {'rows': 50,    'indexes': 2, 'has_text': True},
    'promotions':      {'rows': 100,   'indexes': 3, 'has_text': False},
    'offers':          {'rows': 10,    'indexes': 2, 'has_text': False},
    'cart':            {'rows': 3000,  'indexes': 4, 'has_text': False},
    'orders':          {'rows': 1500,  'indexes': 5, 'has_text': False},
    'claims':          {'rows': 100,   'indexes': 4, 'has_text': True},
    'comments':        {'rows': 200,   'indexes': 3, 'has_text': True},
    'rating':          {'rows': 200,   'indexes': 3, 'has_text': False},
    'favorites':       {'rows': 500,   'indexes': 3, 'has_text': False},
    'returns':         {'rows': 100,   'indexes': 3, 'has_text': True},
    'inbox':           {'rows': 300,   'indexes': 4, 'has_text': True},
    'query_logs':      {'rows': 10000, 'indexes': 1, 'has_text': True},
    'explain_history': {'rows': 1000,  'indexes': 1, 'has_text': True},
}

TEXT_COLUMNS = {
    'products': ['description', 'more_details'],
    'claims': ['reason', 'response'],
    'comments': ['comment'],
    'returns': ['reason', 'response'],
    'inbox': ['text'],
    'query_logs': ['query_text'],
    'explain_history': ['query_text', 'explain_json'],
}

DATE_COLUMNS = {
    'clients': ['birthday'],
    'promotions': ['startdate', 'enddate'],
    'offers': ['startdate', 'enddate'],
    'cart': ['date_cart'],
    'orders': ['orderdate', 'delivereddate'],
    'comments': ['date'],
    'returns': ['date'],
    'inbox': ['date'],
    'query_logs': ['created_at'],
    'explain_history': ['created_at'],
}

FEATURE_ORDER = [
    'query_length_log', 'num_joins', 'num_tables', 'num_where',
    'has_group_by', 'has_order_by', 'has_limit',
    'num_aggregates', 'num_subqueries',
    'target_table_rows', 'rows_examined',
    'has_index_used', 'index_count',
    'involves_text_search', 'involves_date_filter',
    'buffer_pool_hit_ratio', 'connections_count', 'is_peak_hour',
    'joins_per_table', 'complexity_score',
]

# ============================================================
# Variables globales ML
# ============================================================
ml_models = {}
scaler = None
feature_cols = []
model_metrics = {}


# ============================================================
# Entraînement des 3 modèles
# ============================================================

def train_models():
    """Entraîne les 3 modèles au démarrage depuis le dataset CSV."""
    global ml_models, scaler, feature_cols, model_metrics

    csv_path = os.path.join(os.path.dirname(__file__), 'pos_query_performance_dataset.csv')

    # Si le dataset n'existe pas, le générer
    if not os.path.exists(csv_path):
        print("[INFO] Dataset introuvable, génération en cours...")
        try:
            from generate_dataset import main as generate_main
            generate_main()
            # Le fichier est généré dans le répertoire courant
            generated = os.path.join(os.getcwd(), 'sql_query_performance_dataset.csv')
            if os.path.exists(generated) and generated != csv_path:
                import shutil
                shutil.move(generated, csv_path)
        except Exception as e:
            print(f"[ERREUR] Impossible de générer le dataset: {e}")
            print("[INFO] Génération d'un dataset minimal de secours...")
            _generate_fallback_dataset(csv_path)

    if not os.path.exists(csv_path):
        print("[ERREUR] Dataset toujours introuvable:", csv_path)
        return

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != 'is_slow']
    X = df[feature_cols]
    y = df['is_slow']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    ml_models['random_forest'] = rf
    model_metrics['random_forest'] = {
        'accuracy': round(accuracy_score(y_test, y_pred), 3),
        'roc_auc': round(roc_auc_score(y_test, y_proba), 3),
        'f1': round(f1_score(y_test, y_pred), 3),
    }

    # --- XGBoost ---
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=100, max_depth=5, random_state=42,
            eval_metric='logloss', verbosity=0
        )
    else:
        xgb = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1]
    ml_models['xgboost'] = xgb
    model_metrics['xgboost'] = {
        'accuracy': round(accuracy_score(y_test, y_pred), 3),
        'roc_auc': round(roc_auc_score(y_test, y_proba), 3),
        'f1': round(f1_score(y_test, y_pred), 3),
    }

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    y_proba = lr.predict_proba(X_test_scaled)[:, 1]
    ml_models['logistic_regression'] = lr
    model_metrics['logistic_regression'] = {
        'accuracy': round(accuracy_score(y_test, y_pred), 3),
        'roc_auc': round(roc_auc_score(y_test, y_proba), 3),
        'f1': round(f1_score(y_test, y_pred), 3),
    }

    label = "XGBoost" if XGBOOST_AVAILABLE else "GradientBoosting (fallback)"
    print(f"[OK] 3 modèles entraînés (RF, {label}, LR)")
    for name, m in model_metrics.items():
        print(f"     {name}: acc={m['accuracy']} auc={m['roc_auc']} f1={m['f1']}")


def _generate_fallback_dataset(csv_path: str):
    """Génère un petit dataset de secours si generate_dataset échoue."""
    np.random.seed(42)
    n = 1000
    data = {
        'query_length_log': np.random.uniform(3.5, 8.5, n),
        'num_joins': np.random.choice([0,1,2,3,4], n, p=[0.3,0.3,0.2,0.12,0.08]),
        'num_tables': np.random.randint(1, 6, n),
        'num_where': np.random.poisson(3, n).clip(0, 10),
        'has_group_by': np.random.binomial(1, 0.3, n),
        'has_order_by': np.random.binomial(1, 0.4, n),
        'has_limit': np.random.binomial(1, 0.5, n),
        'num_aggregates': np.random.poisson(1, n).clip(0, 5),
        'num_subqueries': np.random.choice([0,1,2,3], n, p=[0.6,0.25,0.1,0.05]),
        'target_table_rows': np.random.choice([50, 200, 1000, 3000, 10000], n),
        'rows_examined': np.random.lognormal(8, 2, n).astype(int).clip(100, 50000000),
        'has_index_used': np.random.binomial(1, 0.6, n),
        'index_count': np.random.poisson(2, n).clip(0, 8),
        'involves_text_search': np.random.binomial(1, 0.15, n),
        'involves_date_filter': np.random.binomial(1, 0.25, n),
        'buffer_pool_hit_ratio': np.random.beta(8, 2, n) * 0.69 + 0.30,
        'connections_count': np.random.poisson(15, n).clip(1, 50),
        'is_peak_hour': np.random.binomial(1, 0.35, n),
    }
    df = pd.DataFrame(data)
    df['joins_per_table'] = np.where(df['num_tables'] > 0, df['num_joins'] / df['num_tables'], 0).round(3)
    df['complexity_score'] = (df['num_joins']*2 + df['num_subqueries']*3 + df['num_aggregates']*1.5 + df['has_group_by']*2 + df['has_order_by']).round(2)

    score = (np.log10(df['rows_examined']+1)*1.2 + (1-df['has_index_used'])*1.5
             + df['num_joins']*1.0 + df['num_subqueries']*1.2
             - df['has_limit']*2.5 - df['has_index_used']*1.2)
    score_norm = (score - score.min()) / (score.max() - score.min())
    df['is_slow'] = (score_norm > np.percentile(score_norm, 55)).astype(int)
    df.to_csv(csv_path, index=False)
    print(f"[OK] Dataset de secours généré: {csv_path} ({n} lignes)")


# ============================================================
# Parsing SQL -> features (adapté au schéma POS)
# ============================================================

def detect_main_table(sql_upper: str) -> str:
    from_match = re.search(r'\bFROM\s+(`?\w+`?)', sql_upper)
    if from_match:
        table = from_match.group(1).strip('`').lower()
        if table in POS_TABLE_INFO:
            return table
    for pattern in [r'\bUPDATE\s+(`?\w+`?)', r'\bINTO\s+(`?\w+`?)', r'\bDELETE\s+FROM\s+(`?\w+`?)']:
        m = re.search(pattern, sql_upper)
        if m:
            table = m.group(1).strip('`').lower()
            if table in POS_TABLE_INFO:
                return table
    for tbl in POS_TABLE_INFO:
        if re.search(r'\b' + tbl.upper() + r'\b', sql_upper):
            return tbl
    return 'products'


def detect_text_search(sql_upper: str, main_table: str) -> int:
    if 'LIKE' in sql_upper:
        if main_table in TEXT_COLUMNS:
            return 1
        for tbl, cols in TEXT_COLUMNS.items():
            if tbl.upper() in sql_upper:
                for col in cols:
                    if col.upper() in sql_upper:
                        return 1
        if '%' in sql_upper:
            return 1
    return 0


def detect_date_filter(sql_upper: str, main_table: str) -> int:
    all_date_cols = set()
    for cols in DATE_COLUMNS.values():
        all_date_cols.update(c.upper() for c in cols)
    for col in all_date_cols:
        if col in sql_upper:
            return 1
    if re.search(r"'\d{4}-\d{2}-\d{2}'", sql_upper):
        return 1
    if re.search(r'\b(CURDATE|NOW|DATE_SUB|DATE_ADD|YEAR|MONTH)\b', sql_upper):
        return 1
    return 0


def parse_sql_to_features(sql_text: str) -> dict:
    """Extrait les features à partir d'une requête SQL sur la BDD POS."""
    sql_upper = sql_text.upper()
    sql_clean = re.sub(r'\s+', ' ', sql_upper).strip()

    main_table = detect_main_table(sql_clean)
    table_info = POS_TABLE_INFO.get(main_table, {'rows': 1000, 'indexes': 2, 'has_text': False})

    query_length_log = round(math.log(max(len(sql_text), 1)), 2)

    join_patterns = re.findall(
        r'\b(INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|CROSS\s+JOIN|NATURAL\s+JOIN|JOIN)\b',
        sql_clean
    )
    num_joins = len(join_patterns)

    num_tables = max(1, num_joins + 1)
    from_match = re.search(r'\bFROM\b(.+?)(?:\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)', sql_clean)
    if from_match:
        from_clause = from_match.group(1)
        from_clause_clean = re.split(r'\b(?:INNER|LEFT|RIGHT|FULL|CROSS|NATURAL)?\s*JOIN\b', from_clause)[0]
        tables_in_from = [t.strip() for t in from_clause_clean.split(',') if t.strip()]
        num_tables = max(len(tables_in_from) + num_joins, 1)
    num_tables = min(num_tables, 8)

    where_match = re.search(r'\bWHERE\b(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)', sql_clean)
    num_where = 0
    if where_match:
        where_clause = where_match.group(1)
        num_where = 1 + len(re.findall(r'\bAND\b|\bOR\b', where_clause))
    num_where = min(num_where, 10)

    has_group_by = 1 if re.search(r'\bGROUP\s+BY\b', sql_clean) else 0
    has_order_by = 1 if re.search(r'\bORDER\s+BY\b', sql_clean) else 0
    has_limit = 1 if re.search(r'\bLIMIT\b', sql_clean) else 0

    aggregates = re.findall(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', sql_clean)
    num_aggregates = min(len(aggregates), 5)

    select_count = len(re.findall(r'\bSELECT\b', sql_clean))
    num_subqueries = min(max(select_count - 1, 0), 3)

    target_table_rows = table_info['rows']
    involves_text_search = detect_text_search(sql_clean, main_table)
    involves_date_filter = detect_date_filter(sql_clean, main_table)

    joins_per_table = round(num_joins / num_tables, 3) if num_tables > 0 else 0
    complexity_score = round(
        num_joins * 2 + num_subqueries * 3 + num_aggregates * 1.5
        + has_group_by * 2 + has_order_by * 1, 2
    )

    return {
        'query_length_log': query_length_log,
        'num_joins': num_joins,
        'num_tables': num_tables,
        'num_where': num_where,
        'has_group_by': has_group_by,
        'has_order_by': has_order_by,
        'has_limit': has_limit,
        'num_aggregates': num_aggregates,
        'num_subqueries': num_subqueries,
        'target_table_rows': target_table_rows,
        'involves_text_search': involves_text_search,
        'involves_date_filter': involves_date_filter,
        'joins_per_table': joins_per_table,
        'complexity_score': complexity_score,
        '_main_table': main_table,
    }


def build_feature_vector(parsed: dict, db_context: dict) -> pd.DataFrame:
    """Combine features parsées + contexte DB pour créer le vecteur complet."""
    row = {}
    for col in FEATURE_ORDER:
        if col in parsed:
            row[col] = parsed[col]
        elif col in db_context:
            row[col] = db_context[col]
        else:
            row[col] = 0
    return pd.DataFrame([row], columns=FEATURE_ORDER)


def predict_all_models(X_df: pd.DataFrame) -> list:
    """Prédit avec les 3 modèles."""
    n = len(X_df)
    all_preds = {}
    all_probas = {}
    for name, model in ml_models.items():
        if name == 'logistic_regression':
            x_input = scaler.transform(X_df)
        else:
            x_input = X_df
        all_preds[name] = model.predict(x_input)
        all_probas[name] = model.predict_proba(x_input)[:, 1]

    results = []
    for idx in range(n):
        row_results = {}
        for name in ml_models:
            pred = int(all_preds[name][idx])
            proba = float(all_probas[name][idx])
            row_results[name] = {
                'prediction': pred,
                'label': 'LENTE' if pred == 1 else 'RAPIDE',
                'probability_slow': round(proba * 100, 1),
            }
        results.append(row_results)
    return results


# ============================================================
# Utilitaires MySQL
# ============================================================

def get_db_connection():
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Erreur connexion MySQL: {err}")
        raise HTTPException(status_code=500, detail=f"Erreur connexion MySQL: {err}")


def execute_query_timed(sql: str):
    """Exécute une requête SQL et retourne les résultats avec le temps."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        start_time = time.time()
        cursor.execute(sql)
        results = cursor.fetchall()
        execution_time = time.time() - start_time
        cursor.close()
        conn.close()
        return {
            "success": True,
            "results": results[:50],  # limiter les résultats retournés
            "execution_time": round(execution_time, 4),
            "row_count": len(results),
            "is_slow": execution_time > SLOW_QUERY_THRESHOLD
        }
    except mysql.connector.Error as err:
        return {
            "success": False,
            "error": str(err),
            "execution_time": 0,
            "row_count": 0,
            "is_slow": False
        }
    except Exception:
        return {
            "success": False,
            "error": "Connexion MySQL indisponible",
            "execution_time": 0,
            "row_count": 0,
            "is_slow": False
        }


def get_table_info():
    """Récupère les informations des tables."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        tables_info = {}
        cursor.execute("SHOW TABLES")
        tables = [row[f'Tables_in_{MYSQL_CONFIG["database"]}'] for row in cursor.fetchall()]
        for table in tables:
            cursor.execute(f"DESCRIBE `{table}`")
            columns = cursor.fetchall()
            cursor.execute(f"SELECT COUNT(*) as count FROM `{table}`")
            count_result = cursor.fetchone()
            count = count_result['count'] if count_result else 0
            cursor.execute(f"""
                SELECT INDEX_NAME, COLUMN_NAME, NON_UNIQUE, INDEX_TYPE
                FROM INFORMATION_SCHEMA.STATISTICS
                WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = '{table}'
                ORDER BY INDEX_NAME, SEQ_IN_INDEX
            """)
            indexes = cursor.fetchall()
            tables_info[table] = {
                "columns": columns,
                "row_count": count,
                "indexes": indexes,
                "index_count": len(indexes)
            }
        cursor.close()
        conn.close()
        return tables_info
    except Exception as e:
        print(f"Erreur récupération tables: {e}")
        return {}


# ============================================================
# Simulateur RL (conservé pour compatibilité frontend)
# ============================================================

class RLSimulator:
    """Simulateur d'agent RL pour l'optimisation d'index."""

    def __init__(self):
        self.current_indexes = 2
        self.performance_history = []
        self.learning_rate = 0.1
        self.exploration_rate = 0.3
        self.max_indexes = RL_CONFIG['max_indexes']

    def get_status(self):
        return {
            'performance': 0.035 + np.random.uniform(-0.01, 0.01),
            'index_count': self.current_indexes,
            'max_indexes': self.max_indexes,
            'status': 'active',
            'message': f'Base optimisée ({self.current_indexes}/{self.max_indexes} index)',
            'agent_mode': 'RL Simulation'
        }

    def optimize(self, steps=5, strategy='balanced'):
        results = []
        for step in range(steps):
            if self.current_indexes < 2 and np.random.random() > 0.3:
                action = 'CREATE'
                self.current_indexes += 1
                reward = 0.8
            elif self.current_indexes > 3 and np.random.random() > 0.5:
                action = 'DROP'
                self.current_indexes -= 1
                reward = 0.4
            else:
                action = 'NOOP'
                reward = 0.1
            self.current_indexes = max(1, min(self.current_indexes, self.max_indexes))
            base_perf = 0.05
            index_benefit = min(self.current_indexes / self.max_indexes, 0.7)
            query_time = base_perf * (1 - index_benefit) + np.random.uniform(-0.005, 0.005)
            query_time = max(0.025, min(query_time, 0.1))
            step_result = {
                'step': step + 1,
                'action': action,
                'reward': round(reward, 4),
                'indexes': self.current_indexes,
                'query_time': round(query_time, 4),
                'explanation': {
                    'CREATE': "Création d'index pour améliorer les lectures",
                    'DROP': "Suppression d'index pour réduire l'overhead",
                    'NOOP': "État optimal, pas de changement"
                }.get(action, '')
            }
            results.append(step_result)
            self.performance_history.append({
                'step': step + 1,
                'query_time': query_time,
                'indexes': self.current_indexes,
                'reward': reward
            })
        return {
            'status': 'success',
            'steps': steps,
            'final_indexes': self.current_indexes,
            'final_query_time': results[-1]['query_time'] if results else 0.035,
            'total_reward': sum(r['reward'] for r in results),
            'steps_details': results,
            'strategy_used': strategy,
            'max_indexes': self.max_indexes
        }

    def get_recommendations(self):
        recommendations = []
        if self.current_indexes < 2:
            recommendations.append({
                'type': 'CREATE', 'priority': 'high',
                'description': 'Index composite sur orders(client_id, orderdate)',
                'sql': 'CREATE INDEX idx_orders_client_date ON orders(client_id, orderdate)',
                'impact': 'Amélioration 40-60% sur les jointures',
                'confidence': 0.85
            })
            recommendations.append({
                'type': 'CREATE', 'priority': 'medium',
                'description': 'Index sur clients(wilaya)',
                'sql': 'CREATE INDEX idx_clients_wilaya ON clients(wilaya)',
                'impact': 'Accélération des filtres géographiques',
                'confidence': 0.75
            })
        elif self.current_indexes >= 4:
            recommendations.append({
                'type': 'DROP', 'priority': 'medium',
                'description': "Réduire les index pour optimiser les écritures",
                'sql': 'DROP INDEX idx_orders_test ON orders',
                'impact': 'Amélioration INSERT/UPDATE de 15-25%',
                'confidence': 0.65
            })
        recommendations.append({
            'type': 'ANALYZE', 'priority': 'low',
            'description': "Analyser l'utilisation des index existants",
            'sql': 'SELECT * FROM information_schema.statistics WHERE table_schema = DATABASE()',
            'impact': 'Identification des index sous-utilisés',
            'confidence': 0.9
        })
        return {
            'current_indexes': self.current_indexes,
            'max_indexes': self.max_indexes,
            'recommendations': recommendations,
            'performance_trend': 'improving' if len(self.performance_history) > 1 and
                self.performance_history[-1]['query_time'] < self.performance_history[0]['query_time']
                else 'stable'
        }

    def get_learning_stats(self):
        if len(self.performance_history) < 2:
            return {
                'total_steps': len(self.performance_history),
                'average_reward': 0,
                'performance_improvement': 0,
                'learning_progress': 0
            }
        rewards = [h['reward'] for h in self.performance_history]
        query_times = [h['query_time'] for h in self.performance_history]
        return {
            'total_steps': len(self.performance_history),
            'average_reward': round(np.mean(rewards), 4),
            'performance_improvement': round((query_times[0] - query_times[-1]) / query_times[0] * 100, 1),
            'learning_progress': min(len(self.performance_history) / 100, 1.0),
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate
        }


# ============================================================
# Pydantic models
# ============================================================

class QueryRequest(BaseModel):
    question: str
    user_id: str = "default"

class SQLAnalysisRequest(BaseModel):
    sql: str
    # Contexte DB optionnel
    rows_examined: Optional[int] = None
    has_index_used: Optional[int] = None
    index_count: Optional[int] = None
    buffer_pool_hit_ratio: Optional[float] = None
    connections_count: Optional[int] = None
    is_peak_hour: Optional[int] = None

class OptimizeRequest(BaseModel):
    steps: int = 5
    strategy: str = "balanced"


# ============================================================
# INITIALISATION
# ============================================================

app = FastAPI(
    title="SADOP API v5.0 - ML Réel",
    description="Prédiction de performance SQL avec Random Forest, XGBoost et Logistic Regression + RL",
    version="5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rl_agent = RLSimulator()

# Entraîner les modèles au démarrage
print("=" * 60)
print("🚀 SADOP API v5.0 - Entraînement des modèles ML réels")
print("=" * 60)
train_models()


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "SADOP API v5.0",
        "description": "Prédiction ML réelle avec 3 algorithmes + RL simulation",
        "models": list(ml_models.keys()),
        "model_metrics": model_metrics,
        "xgboost_native": XGBOOST_AVAILABLE,
        "features_count": len(FEATURE_ORDER),
        "endpoints": {
            "GET /": "Cette page",
            "GET /health": "État du service",
            "POST /api/analyze/sql": "Analyse SQL avec 3 modèles ML",
            "POST /predict/sql": "Prédiction depuis SQL brut",
            "POST /predict/form": "Prédiction depuis features manuelles",
            "POST /predict/csv": "Prédiction depuis CSV uploadé",
            "POST /chat": "Chat avec l'agent IA",
            "GET /api/rl/status": "Statut RL",
            "GET /api/rl/recommendations": "Recommandations RL",
            "POST /api/rl/optimize": "Optimisation RL",
            "GET /api/tables": "Tables de la BDD",
            "GET /api/metrics": "Métriques des modèles ML",
        }
    }


@app.get("/health")
async def health_check():
    db_status = "disconnected"
    db_name = "unknown"
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.execute("SELECT DATABASE()")
        db_name = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    rl_status = rl_agent.get_status()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": db_status,
        "database_name": db_name,
        "ml_models": list(ml_models.keys()),
        "ml_models_count": len(ml_models),
        "xgboost": "native" if XGBOOST_AVAILABLE else "fallback",
        "rl_agent": rl_status['status'],
        "rl_indexes": f"{rl_status['index_count']}/{rl_status['max_indexes']}",
        "version": "5.0",
        "slow_query_threshold": SLOW_QUERY_THRESHOLD
    }


# ==================== ML ENDPOINTS ====================

@app.post("/api/analyze/sql")
async def analyze_sql_endpoint(request: SQLAnalysisRequest):
    """Analyse une requête SQL avec les 3 vrais modèles ML."""
    try:
        parsed = parse_sql_to_features(request.sql)
        main_table = parsed.pop('_main_table')
        table_info = POS_TABLE_INFO.get(main_table, {})

        db_context = {
            'target_table_rows': parsed.get('target_table_rows', table_info.get('rows', 1000)),
            'rows_examined': request.rows_examined or (parsed.get('target_table_rows', 1000) * 2),
            'has_index_used': request.has_index_used if request.has_index_used is not None else 1,
            'index_count': request.index_count or table_info.get('indexes', 2),
            'buffer_pool_hit_ratio': request.buffer_pool_hit_ratio or 0.85,
            'connections_count': request.connections_count or 12,
            'is_peak_hour': request.is_peak_hour or 0,
        }

        X_df = build_feature_vector(parsed, db_context)
        all_results = predict_all_models(X_df)
        model_preds = all_results[0]  # un seul SQL

        # Résultat principal (XGBoost par défaut pour la compatibilité frontend)
        xgb_result = model_preds.get('xgboost', model_preds.get('random_forest', {}))
        is_slow = xgb_result.get('prediction', 0) == 1
        slow_proba = xgb_result.get('probability_slow', 50.0) / 100.0

        # Raisons de la prédiction
        reasons = []
        if parsed.get('num_joins', 0) > 0:
            reasons.append(f"{parsed['num_joins']} jointure(s)")
        if parsed.get('has_group_by', 0):
            reasons.append("GROUP BY détecté")
        if parsed.get('num_subqueries', 0) > 0:
            reasons.append(f"{parsed['num_subqueries']} sous-requête(s)")
        if parsed.get('num_aggregates', 0) > 0:
            reasons.append(f"{parsed['num_aggregates']} agrégat(s)")
        if not db_context.get('has_index_used', 1):
            reasons.append("Pas d'index utilisé")
        if db_context.get('rows_examined', 0) > 10000:
            reasons.append(f"Beaucoup de lignes examinées ({db_context['rows_examined']:,})")

        # Exécution réelle sur MySQL
        execution = execute_query_timed(request.sql)

        # Format compatible avec le frontend existant
        prediction = {
            "is_slow": is_slow,
            "slow_probability": round(slow_proba, 3),
            "fast_probability": round(1 - slow_proba, 3),
            "confidence": round(abs(slow_proba - 0.5) * 2, 3),
            "reasons": reasons,
            "complexity_score": parsed.get('complexity_score', 0),
            "threshold": SLOW_QUERY_THRESHOLD
        }

        return {
            "success": True,
            "sql": request.sql,
            "prediction": prediction,
            "all_models": model_preds,
            "execution": execution,
            "parsed_features": parsed,
            "db_context": db_context,
            "detected_table": main_table,
            "slow_threshold": SLOW_QUERY_THRESHOLD
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/predict/sql")
async def predict_sql(request: SQLAnalysisRequest):
    """Prédiction depuis une requête SQL brute (endpoint original Flask adapté)."""
    return await analyze_sql_endpoint(request)


@app.post("/predict/form")
async def predict_form(data: dict):
    """Prédiction depuis les features manuelles."""
    try:
        row = {}
        for col in FEATURE_ORDER:
            val = data.get(col, 0)
            row[col] = float(val)
        if row['num_tables'] > 0:
            row['joins_per_table'] = round(row['num_joins'] / row['num_tables'], 3)
        row['complexity_score'] = round(
            row['num_joins'] * 2 + row['num_subqueries'] * 3
            + row['num_aggregates'] * 1.5 + row['has_group_by'] * 2
            + row['has_order_by'] * 1, 2
        )
        X_df = pd.DataFrame([row], columns=FEATURE_ORDER)
        results = predict_all_models(X_df)
        return {'success': True, 'predictions': results, 'features': row}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """Prédiction depuis un fichier CSV uploadé."""
    try:
        if not file.filename.lower().endswith('.csv'):
            return {'success': False, 'error': 'Le fichier doit être un CSV'}

        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        missing_cols = [c for c in FEATURE_ORDER if c not in df.columns]
        if 'joins_per_table' in missing_cols and 'num_joins' in df.columns and 'num_tables' in df.columns:
            df['joins_per_table'] = np.where(df['num_tables'] > 0, df['num_joins'] / df['num_tables'], 0).round(3)
            missing_cols.remove('joins_per_table')
        if 'complexity_score' in missing_cols and all(
            c in df.columns for c in ['num_joins', 'num_subqueries', 'num_aggregates', 'has_group_by', 'has_order_by']
        ):
            df['complexity_score'] = (
                df['num_joins'] * 2 + df['num_subqueries'] * 3
                + df['num_aggregates'] * 1.5 + df['has_group_by'] * 2
                + df['has_order_by'] * 1
            ).round(2)
            missing_cols.remove('complexity_score')
        if missing_cols:
            return {'success': False, 'error': f'Colonnes manquantes: {", ".join(missing_cols)}'}

        X_df = df[FEATURE_ORDER].copy()
        has_target = 'is_slow' in df.columns
        results = predict_all_models(X_df)

        response_rows = []
        for i, row_preds in enumerate(results):
            row_data = {'index': i, 'features': X_df.iloc[i].to_dict(), 'predictions': row_preds}
            if has_target:
                row_data['actual'] = int(df['is_slow'].iloc[i])
            response_rows.append(row_data)

        csv_metrics = None
        if has_target:
            y_true = df['is_slow'].values
            csv_metrics = {}
            for name in ml_models:
                x_input = scaler.transform(X_df) if name == 'logistic_regression' else X_df
                y_pred = ml_models[name].predict(x_input)
                y_proba = ml_models[name].predict_proba(x_input)[:, 1]
                csv_metrics[name] = {
                    'accuracy': round(accuracy_score(y_true, y_pred), 3),
                    'roc_auc': round(roc_auc_score(y_true, y_proba), 3),
                    'f1': round(f1_score(y_true, y_pred), 3),
                }

        return {
            'success': True,
            'total_rows': len(X_df),
            'rows': response_rows[:200],
            'truncated': len(X_df) > 200,
            'csv_metrics': csv_metrics,
            'has_target': has_target,
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


@app.get("/api/metrics")
async def get_model_metrics():
    """Retourne les métriques des 3 modèles ML."""
    return {
        "success": True,
        "models": model_metrics,
        "xgboost_native": XGBOOST_AVAILABLE,
        "features": FEATURE_ORDER,
        "pos_tables": list(POS_TABLE_INFO.keys())
    }


@app.get("/table-info/{table_name}")
async def table_info_endpoint(table_name: str):
    """Retourne les infos d'une table POS."""
    info = POS_TABLE_INFO.get(table_name)
    if not info:
        raise HTTPException(status_code=404, detail="Table inconnue")
    return {
        'table': table_name,
        'rows': info['rows'],
        'indexes': info['indexes'],
        'has_text': info['has_text'],
        'text_columns': TEXT_COLUMNS.get(table_name, []),
        'date_columns': DATE_COLUMNS.get(table_name, []),
    }


# ==================== CHAT ENDPOINT ====================

@app.post("/chat")
async def chat_with_agent(request: QueryRequest):
    """Chat avec l'agent — analyse SQL si détecté, sinon info générale."""
    try:
        input_lower = request.question.lower()

        # Détection SQL
        if any(kw in input_lower for kw in ['select', 'insert', 'update', 'delete', 'from ', 'where ']):
            # C'est une requête SQL → analyser
            sql_req = SQLAnalysisRequest(sql=request.question)
            analysis = await analyze_sql_endpoint(sql_req)
            if analysis.get("success"):
                pred = analysis["prediction"]
                models_detail = analysis.get("all_models", {})
                lines = [
                    "## 🔍 Analyse SQL (3 modèles ML réels)",
                    f"```sql\n{request.question}\n```",
                    f"**Table détectée:** {analysis.get('detected_table', '?')}",
                    "",
                ]
                for mname, mresult in models_detail.items():
                    icon = "🔴" if mresult['prediction'] == 1 else "🟢"
                    lines.append(f"{icon} **{mname}**: {mresult['label']} ({mresult['probability_slow']}% slow)")
                if pred["reasons"]:
                    lines.append("\n**Raisons:**")
                    for r in pred["reasons"]:
                        lines.append(f"- {r}")
                exe = analysis.get("execution", {})
                if exe.get("success"):
                    lines.append(f"\n⚡ **Exécution réelle:** {exe['execution_time']:.3f}s ({exe['row_count']} lignes)")
                return {"success": True, "question": request.question, "response": "\n".join(lines),
                        "source": "SADOP ML (RF + XGBoost + LR)", "timestamp": datetime.now().isoformat()}
            else:
                return {"success": True, "question": request.question,
                        "response": f"❌ Erreur analyse: {analysis.get('error', '?')}",
                        "source": "SADOP", "timestamp": datetime.now().isoformat()}

        elif any(kw in input_lower for kw in ['optimiser', 'optimize', 'rl', 'index']):
            result = rl_agent.optimize(steps=5)
            lines = [
                "## 🤖 Optimisation RL",
                f"**Étapes:** {result['steps']} | **Index finaux:** {result['final_indexes']}/{result['max_indexes']}",
                f"**Performance:** {result['final_query_time']:.3f}s | **Récompense:** {result['total_reward']:.3f}",
            ]
            for s in result['steps_details']:
                lines.append(f"- Étape {s['step']}: {s['action']} (reward: {s['reward']:.3f})")
            return {"success": True, "question": request.question, "response": "\n".join(lines),
                    "source": "SADOP RL", "timestamp": datetime.now().isoformat()}

        elif any(kw in input_lower for kw in ['métrique', 'metric', 'performance', 'accuracy', 'précision']):
            lines = ["## 📊 Métriques des modèles ML", ""]
            for name, m in model_metrics.items():
                lines.append(f"**{name}**: Accuracy={m['accuracy']} | AUC={m['roc_auc']} | F1={m['f1']}")
            return {"success": True, "question": request.question, "response": "\n".join(lines),
                    "source": "SADOP ML", "timestamp": datetime.now().isoformat()}

        else:
            return {
                "success": True,
                "question": request.question,
                "response": f"""## 🤖 Assistant SADOP v5.0

**Modèles ML actifs:** {', '.join(ml_models.keys())}

**Commandes:**
- Envoyez une requête SQL pour l'analyser avec 3 modèles
- "optimiser" → lancer l'optimisation RL
- "métriques" → voir les performances des modèles
- "tables" → structure de la BDD

**Exemple:** `SELECT * FROM orders JOIN clients ON orders.client_id = clients.id`""",
                "source": "SADOP", "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ==================== RL ENDPOINTS ====================

@app.get("/api/rl/status")
async def get_rl_status():
    try:
        status = rl_agent.get_status()
        stats = rl_agent.get_learning_stats()
        return {"success": True, "data": {"status": status, "learning_stats": stats, "configuration": RL_CONFIG}}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/rl/recommendations")
async def get_rl_recommendations():
    try:
        return {"success": True, "data": rl_agent.get_recommendations()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/rl/optimize")
async def optimize_with_rl(request: OptimizeRequest):
    try:
        result = rl_agent.optimize(steps=request.steps, strategy=request.strategy)
        return {"success": True, "data": result, "message": f"Optimisation RL terminée ({request.steps} étapes)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/rl/learning-stats")
async def get_rl_learning_stats():
    try:
        return {"success": True, "data": rl_agent.get_learning_stats()}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ==================== DB ENDPOINTS ====================

@app.get("/api/tables")
async def get_tables():
    try:
        tables_info = get_table_info()
        return {"success": True, "data": tables_info, "count": len(tables_info)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/performance")
async def get_performance_summary():
    try:
        rl_status = rl_agent.get_status()
        learning_stats = rl_agent.get_learning_stats()
        return {
            "success": True,
            "data": {
                "rl_status": rl_status,
                "learning_stats": learning_stats,
                "ml_metrics": model_metrics,
                "slow_threshold": SLOW_QUERY_THRESHOLD,
                "max_indexes": RL_CONFIG['max_indexes']
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ==================== LANCEMENT ====================
if __name__ == "__main__":
    print("=" * 60)
    print(f"🔮 ML: {len(ml_models)} modèles ({', '.join(ml_models.keys())})")
    print(f"🤖 RL: Simulation active")
    print(f"🌐 URL: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
