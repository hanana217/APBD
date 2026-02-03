import pandas as pd
import mysql.connector
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

# ===============================
# DB CONFIG
# ===============================
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3308,
    'user': 'apbd_user',
    'password': 'apbd_pass',
    'database': 'pos'
}

# ===============================
# LOAD DATA
# ===============================
conn = mysql.connector.connect(**DB_CONFIG)
df = pd.read_sql("SELECT * FROM dataset_ml", conn)
conn.close()

print("Dataset brut:", df.shape)

# ===============================
# NETTOYAGE
# ===============================
# Garder toutes les colonnes sauf sql_text et explain_text pour le moment
df_original = df.copy()
df = df.dropna(subset=['is_slow', 'sql_text'])

# Remplir les valeurs manquantes
numerical_cols = ['num_joins', 'num_where', 'has_order_by', 'has_group_by',
                  'query_length', 'rows_examined', 'using_filesort',
                  'using_temporary', 'cpu_usage', 'buffer_pool_hit_ratio',
                  'connections_count', 'hour_of_day', 'day_of_week',
                  'execution_time', 'index_count']

categorical_cols = ['access_type', 'key_used']

for col in numerical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)
        
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna("UNKNOWN")

# ===============================
# ENCODAGE CATÉGORIEL
# ===============================
label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# ===============================
# GÉNÉRATION DES EMBEDDINGS SQL
# ===============================
print("⏳ Génération des embeddings SQL...")

model = SentenceTransformer("all-MiniLM-L6-v2")

df['sql_text'] = df['sql_text'].fillna("").astype(str)
sql_texts = df['sql_text'].tolist()

sql_embeddings = model.encode(
    sql_texts,
    batch_size=32,
    show_progress_bar=True
)

sql_embeddings = np.array(sql_embeddings)
print(f"✅ Embeddings SQL shape: {sql_embeddings.shape}")

# ===============================
# CRÉATION DU DATASET COMPLET
# ===============================
# Créer un DataFrame avec toutes les features
df_all_features = df.copy()

# Ajouter les embeddings comme nouvelles colonnes
for i in range(sql_embeddings.shape[1]):
    df_all_features[f'sql_emb_{i}'] = sql_embeddings[:, i]

# Sauvegarder aussi les textes SQL et explain pour référence
df_all_features['sql_text_original'] = df_original['sql_text']
df_all_features['explain_text'] = df_original['explain_text']

# ===============================
# SAUVEGARDE
# ===============================
df_all_features.to_csv("../data/exports/dataset_with_embeddings.csv", index=False)

print("✅ Dataset complet avec embeddings prêt :", df_all_features.shape)
print(f"📊 Fichier sauvegardé : ../data/exports/dataset_with_embeddings.csv")
print(f"📋 Colonnes : {df_all_features.columns.tolist()}")