"""
Application Web Flask - Prediction de Performance SQL (BDD POS)
================================================================
Interface web pour predire si une requete SQL sur la BDD POS sera
lente ou rapide en utilisant 3 algorithmes:
  Random Forest, XGBoost, Logistic Regression.

Schema POS: admin, clients, wilayas, products, promotions, offers,
            cart, orders, claims, comments, rating, favorites,
            returns, inbox, query_logs, explain_history
"""

import os
import re
import math
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ============================================================
# Metadata POS (meme que dans generate_dataset.py)
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

# Colonnes TEXT/LONGTEXT par table
TEXT_COLUMNS = {
    'products': ['description', 'more_details'],
    'claims': ['reason', 'response'],
    'comments': ['comment'],
    'returns': ['reason', 'response'],
    'inbox': ['text'],
    'query_logs': ['query_text'],
    'explain_history': ['query_text', 'explain_json'],
}

# Colonnes DATE par table
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

# ============================================================
# Variables globales
# ============================================================
models = {}
scaler = None
feature_cols = []
model_metrics = {}

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


def train_models():
    """Entraine les 3 modeles au demarrage depuis le dataset CSV."""
    global models, scaler, feature_cols, model_metrics

    csv_path = os.path.join(os.path.dirname(__file__), 'data/exports/dataset_final.csv')
    if not os.path.exists(csv_path):
        print("[ERREUR] Dataset introuvable:", csv_path)
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
    models['random_forest'] = rf
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
    models['xgboost'] = xgb
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
    models['logistic_regression'] = lr
    model_metrics['logistic_regression'] = {
        'accuracy': round(accuracy_score(y_test, y_pred), 3),
        'roc_auc': round(roc_auc_score(y_test, y_proba), 3),
        'f1': round(f1_score(y_test, y_pred), 3),
    }

    label = "XGBoost" if XGBOOST_AVAILABLE else "GradientBoosting (fallback)"
    print(f"[OK] 3 modeles entraines (RF, {label}, LR)")
    for name, m in model_metrics.items():
        print(f"     {name}: acc={m['accuracy']} auc={m['roc_auc']} f1={m['f1']}")


# ============================================================
# Parsing SQL -> features (adapte au schema POS)
# ============================================================

def detect_main_table(sql_upper: str) -> str:
    """Detecte la table principale dans la requete SQL POS."""
    # Chercher dans FROM
    from_match = re.search(
        r'\bFROM\s+(`?\w+`?)', sql_upper
    )
    if from_match:
        table = from_match.group(1).strip('`').lower()
        if table in POS_TABLE_INFO:
            return table

    # Chercher dans UPDATE / INSERT INTO / DELETE FROM
    for pattern in [r'\bUPDATE\s+(`?\w+`?)', r'\bINTO\s+(`?\w+`?)', r'\bDELETE\s+FROM\s+(`?\w+`?)']:
        m = re.search(pattern, sql_upper)
        if m:
            table = m.group(1).strip('`').lower()
            if table in POS_TABLE_INFO:
                return table

    # Chercher n'importe quelle table POS mentionnee
    for tbl in POS_TABLE_INFO:
        if re.search(r'\b' + tbl.upper() + r'\b', sql_upper):
            return tbl

    return 'products'  # default


def detect_text_search(sql_upper: str, main_table: str) -> int:
    """Detecte si la requete fait un LIKE sur une colonne TEXT."""
    if 'LIKE' in sql_upper:
        # Verifier si la table a des colonnes TEXT
        if main_table in TEXT_COLUMNS:
            return 1
        # Verifier les tables jointes
        for tbl, cols in TEXT_COLUMNS.items():
            if tbl.upper() in sql_upper:
                for col in cols:
                    if col.upper() in sql_upper:
                        return 1
        # LIKE generique
        if '%' in sql_upper:
            return 1
    return 0


def detect_date_filter(sql_upper: str, main_table: str) -> int:
    """Detecte si la requete filtre sur des colonnes DATE."""
    all_date_cols = set()
    for cols in DATE_COLUMNS.values():
        all_date_cols.update(c.upper() for c in cols)

    for col in all_date_cols:
        if col in sql_upper:
            return 1

    # Patterns de date dans WHERE
    if re.search(r"'\d{4}-\d{2}-\d{2}'", sql_upper):
        return 1
    if re.search(r'\b(CURDATE|NOW|DATE_SUB|DATE_ADD|YEAR|MONTH)\b', sql_upper):
        return 1

    return 0


def parse_sql_to_features(sql_text: str) -> dict:
    """
    Extrait les features a partir d'une requete SQL sur la BDD POS.
    """
    sql_upper = sql_text.upper()
    sql_clean = re.sub(r'\s+', ' ', sql_upper).strip()

    # Table principale
    main_table = detect_main_table(sql_clean)
    table_info = POS_TABLE_INFO.get(main_table, {'rows': 1000, 'indexes': 2, 'has_text': False})

    # query_length_log
    query_length_log = round(math.log(max(len(sql_text), 1)), 2)

    # num_joins
    join_patterns = re.findall(
        r'\b(INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|CROSS\s+JOIN|NATURAL\s+JOIN|JOIN)\b',
        sql_clean
    )
    num_joins = len(join_patterns)

    # num_tables
    num_tables = max(1, num_joins + 1)
    from_match = re.search(r'\bFROM\b(.+?)(?:\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)', sql_clean)
    if from_match:
        from_clause = from_match.group(1)
        from_clause_clean = re.split(r'\b(?:INNER|LEFT|RIGHT|FULL|CROSS|NATURAL)?\s*JOIN\b', from_clause)[0]
        tables_in_from = [t.strip() for t in from_clause_clean.split(',') if t.strip()]
        num_tables = max(len(tables_in_from) + num_joins, 1)
    num_tables = min(num_tables, 8)

    # num_where
    where_match = re.search(r'\bWHERE\b(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)', sql_clean)
    num_where = 0
    if where_match:
        where_clause = where_match.group(1)
        num_where = 1 + len(re.findall(r'\bAND\b|\bOR\b', where_clause))
    num_where = min(num_where, 10)

    # Clauses
    has_group_by = 1 if re.search(r'\bGROUP\s+BY\b', sql_clean) else 0
    has_order_by = 1 if re.search(r'\bORDER\s+BY\b', sql_clean) else 0
    has_limit = 1 if re.search(r'\bLIMIT\b', sql_clean) else 0

    # Aggregates
    aggregates = re.findall(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', sql_clean)
    num_aggregates = min(len(aggregates), 5)

    # Subqueries
    select_count = len(re.findall(r'\bSELECT\b', sql_clean))
    num_subqueries = min(max(select_count - 1, 0), 3)

    # POS-specific
    target_table_rows = table_info['rows']
    involves_text_search = detect_text_search(sql_clean, main_table)
    involves_date_filter = detect_date_filter(sql_clean, main_table)

    # Derived
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
    """
    Combine features parsees + contexte DB pour creer le vecteur complet.
    """
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
    """Predit avec les 3 modeles."""
    results = []
    for idx in range(len(X_df)):
        row_results = {}
        x_row = X_df.iloc[[idx]]

        for name, model in models.items():
            if name == 'logistic_regression':
                x_input = scaler.transform(x_row)
            else:
                x_input = x_row

            pred = int(model.predict(x_input)[0])
            proba = float(model.predict_proba(x_input)[0][1])
            row_results[name] = {
                'prediction': pred,
                'label': 'LENTE' if pred == 1 else 'RAPIDE',
                'probability_slow': round(proba * 100, 1),
            }
        results.append(row_results)
    return results


# ============================================================
# Routes Flask
# ============================================================

@app.route('/')
def index():
    return render_template('index.html', metrics=model_metrics,
                           xgboost_available=XGBOOST_AVAILABLE,
                           pos_tables=list(POS_TABLE_INFO.keys()))


@app.route('/predict/form', methods=['POST'])
def predict_form():
    """Prediction depuis le formulaire manuel."""
    try:
        data = request.get_json()
        row = {}
        for col in FEATURE_ORDER:
            val = data.get(col, 0)
            row[col] = float(val)

        # Recalculer derivees
        if row['num_tables'] > 0:
            row['joins_per_table'] = round(row['num_joins'] / row['num_tables'], 3)
        row['complexity_score'] = round(
            row['num_joins'] * 2 + row['num_subqueries'] * 3
            + row['num_aggregates'] * 1.5 + row['has_group_by'] * 2
            + row['has_order_by'] * 1, 2
        )

        X_df = pd.DataFrame([row], columns=FEATURE_ORDER)
        results = predict_all_models(X_df)
        return jsonify({'success': True, 'predictions': results, 'features': row})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/predict/sql', methods=['POST'])
def predict_sql():
    """Prediction depuis une requete SQL brute + contexte DB."""
    try:
        data = request.get_json()
        sql_text = data.get('sql', '').strip()
        if not sql_text:
            return jsonify({'success': False, 'error': 'Requete SQL vide'})

        parsed = parse_sql_to_features(sql_text)
        main_table = parsed.pop('_main_table')

        # Contexte DB
        table_info = POS_TABLE_INFO.get(main_table, {})
        db_context = {
            'target_table_rows': int(data.get('target_table_rows', parsed.get('target_table_rows', table_info.get('rows', 1000)))),
            'rows_examined': int(data.get('rows_examined', parsed.get('target_table_rows', 1000) * 2)),
            'has_index_used': int(data.get('has_index_used', 1)),
            'index_count': int(data.get('index_count', table_info.get('indexes', 2))),
            'buffer_pool_hit_ratio': float(data.get('buffer_pool_hit_ratio', 0.85)),
            'connections_count': int(data.get('connections_count', 12)),
            'is_peak_hour': int(data.get('is_peak_hour', 0)),
        }

        X_df = build_feature_vector(parsed, db_context)
        results = predict_all_models(X_df)
        features_used = X_df.iloc[0].to_dict()

        return jsonify({
            'success': True,
            'predictions': results,
            'parsed_features': parsed,
            'db_context': db_context,
            'features': features_used,
            'detected_table': main_table,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/predict/csv', methods=['POST'])
def predict_csv():
    """Prediction depuis un fichier CSV uploade."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Aucun fichier envoye'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Aucun fichier selectionne'})

        if not file.filename.lower().endswith('.csv'):
            return jsonify({'success': False, 'error': 'Le fichier doit etre un CSV'})

        df = pd.read_csv(file)

        missing_cols = [c for c in FEATURE_ORDER if c not in df.columns]

        # Calculer derivees si manquantes
        if 'joins_per_table' in missing_cols and 'num_joins' in df.columns and 'num_tables' in df.columns:
            df['joins_per_table'] = np.where(
                df['num_tables'] > 0, df['num_joins'] / df['num_tables'], 0
            ).round(3)
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
            return jsonify({
                'success': False,
                'error': f'Colonnes manquantes: {", ".join(missing_cols)}'
            })

        X_df = df[FEATURE_ORDER].copy()
        has_target = 'is_slow' in df.columns

        results = predict_all_models(X_df)

        response_rows = []
        for i, row_preds in enumerate(results):
            row_data = {
                'index': i,
                'features': X_df.iloc[i].to_dict(),
                'predictions': row_preds,
            }
            if has_target:
                row_data['actual'] = int(df['is_slow'].iloc[i])
            response_rows.append(row_data)

        csv_metrics = None
        if has_target:
            y_true = df['is_slow'].values
            csv_metrics = {}
            for name in models:
                if name == 'logistic_regression':
                    x_input = scaler.transform(X_df)
                else:
                    x_input = X_df
                y_pred = models[name].predict(x_input)
                y_proba = models[name].predict_proba(x_input)[:, 1]
                csv_metrics[name] = {
                    'accuracy': round(accuracy_score(y_true, y_pred), 3),
                    'roc_auc': round(roc_auc_score(y_true, y_proba), 3),
                    'f1': round(f1_score(y_true, y_pred), 3),
                }

        return jsonify({
            'success': True,
            'total_rows': len(X_df),
            'rows': response_rows[:200],
            'truncated': len(X_df) > 200,
            'csv_metrics': csv_metrics,
            'has_target': has_target,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/table-info/<table_name>')
def table_info(table_name):
    """Retourne les infos d'une table POS."""
    info = POS_TABLE_INFO.get(table_name)
    if not info:
        return jsonify({'error': 'Table inconnue'}), 404
    return jsonify({
        'table': table_name,
        'rows': info['rows'],
        'indexes': info['indexes'],
        'has_text': info['has_text'],
        'text_columns': TEXT_COLUMNS.get(table_name, []),
        'date_columns': DATE_COLUMNS.get(table_name, []),
    })


# ============================================================
# Demarrage
# ============================================================

print("[*] Entrainement des modeles sur le dataset POS...")
train_models()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
