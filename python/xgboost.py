import pandas as pd
import mysql.connector
import os

# -----------------------
# CONFIGURATION BASE DE DONNÉES
# -----------------------
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3308,
    'user': 'apbd_user',
    'password': 'apbd_pass',
    'database': 'pos'
}

EXPORT_DIR = "../data/exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# -----------------------
# FONCTION DE CRÉATION DU CSV
# -----------------------
def generate_dataset_csv():
    print("="*60)
    print("📊 GENERATION DU DATASET CSV À PARTIR DE LA TABLE 'dataset'")
    print("="*60)
    
    # Connexion à la base de données
    conn = mysql.connector.connect(**DB_CONFIG)
    query = "SELECT * FROM dataset_final_balanced"
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Nettoyage de base
    df = df.dropna(subset=['is_slow'])  # supprimer les lignes sans target
    df = df.fillna(0)  # remplacer les valeurs manquantes par 0
    
    print(f"🧹 Après nettoyage: {df.shape[0]} lignes restantes")
    
    # Features à conserver pour XGBoost
    features = [
        'num_joins', 'num_where',
        'has_order_by', 'has_group_by',
        'query_length', 'rows_examined',
        'using_filesort', 'using_temporary',
        'cpu_usage', 'buffer_pool_hit_ratio',
        'connections_count', 'hour_of_day', 'day_of_week',
        'execution_time', 'index_count', 'index_coverage',
        'query_complexity_score', 'num_subqueries',
        'num_predicates', 'has_like', 'in_list_size',
        'num_aggregates', 'estimated_row', 'is_peak_hour'
    ]
    if 'rows_examined' not in df.columns:
        print("ℹ️ Colonne 'rows_examined' manquante, génération artificielle...")
        df['rows_examined'] = df['num_joins'] * 100  # ou une autre approximation

    # Filtrer uniquement les colonnes existantes
    features = [col for col in features if col in df.columns]
    features.append('is_slow')  # ajouter la target
    
    df_xgb = df[features].copy()
    
    # Sauvegarde CSV
    csv_path = os.path.join(EXPORT_DIR, "dataset_final.csv")
    df_xgb.to_csv(csv_path, index=False)
    
    print(f"✅ Dataset CSV généré: {csv_path}")
    print(f"📊 Dimensions finales: {df_xgb.shape}")
    
    return df_xgb

# -----------------------
# EXECUTION
# -----------------------
if __name__ == "__main__":
    df_csv = generate_dataset_csv()
