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
# FEATURES
# ===============================
numerical_features = [
    'num_joins', 'num_where', 'has_order_by', 'has_group_by',
    'query_length', 'rows_examined', 'using_filesort',
    'using_temporary', 'index_count',
    'buffer_pool_hit_ratio', 'connections_count',
    'hour_of_day', 'day_of_week'
]

categorical_features = ['access_type']

target_classification = 'is_slow'
target_regression = 'execution_time'

# ===============================
# CLEANING
# ===============================
df = df.dropna(subset=[target_classification, 'sql_text'])

df[numerical_features] = df[numerical_features].fillna(0)
df[categorical_features] = df[categorical_features].fillna("UNKNOWN")

# ===============================
# LABEL ENCODING (categorical)
# ===============================
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ===============================
# SQL TEXT EMBEDDING
# ===============================
print("⏳ Génération des embeddings SQL...")

model = SentenceTransformer("all-MiniLM-L6-v2")

sql_texts = df['sql_text'].astype(str).tolist()
sql_embeddings = model.encode(
    sql_texts,
    batch_size=32,
    show_progress_bar=True
)

sql_embeddings = np.array(sql_embeddings)
print("Embeddings SQL shape:", sql_embeddings.shape)  # (N, 384)

# ===============================
# DATASET FINAL
# ===============================
X_numeric = df[numerical_features + categorical_features].values

X_final = np.hstack([X_numeric, sql_embeddings])

# Create column names
embedding_cols = [f"sql_emb_{i}" for i in range(sql_embeddings.shape[1])]
final_columns = (
    numerical_features +
    categorical_features +
    embedding_cols
)

df_final = pd.DataFrame(X_final, columns=final_columns)

df_final[target_classification] = df[target_classification].values
df_final[target_regression] = df[target_regression].values

# ===============================
# SAVE
# ===============================
df_final.to_csv("../data/exports/dataset_sql_embedding_ready.csv", index=False)

print("✅ Dataset Embedding SQL prêt :", df_final.shape)
