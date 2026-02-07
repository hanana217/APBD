import pandas as pd
import mysql.connector
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

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
print("📥 Chargement des données depuis MySQL...")
conn = mysql.connector.connect(**DB_CONFIG)
df = pd.read_sql("SELECT * FROM dataset_final", conn)
conn.close()

print(f"Dataset chargé : {df.shape}")
print(f"Colonnes disponibles : {df.columns.tolist()}")

# ===============================
# PRÉPARATION DES DONNÉES
# ===============================
print("\n🧹 Nettoyage des données...")

# Conserver une copie des textes SQL originaux
df_original = df.copy()

# Vérifier la colonne cible
if 'is_slow' not in df.columns:
    raise ValueError("La colonne 'is_slow' est requise mais n'existe pas dans la table")

# Remplir les valeurs manquantes pour les colonnes numériques
numerical_cols = [
    'query_length', 'query_length_log', 'num_joins', 'num_where',
    'num_subqueries', 'num_predicates', 'num_aggregates', 'num_functions',
    'num_tables', 'max_in_list_size', 'hour_of_day', 'day_of_week',
    'connections_count', 'buffer_pool_hit_ratio', 'estimated_index_count',
    'estimated_table_size_mb', 'joins_per_table', 'predicates_per_where',
    'complexity_density'
]

# Filtrer pour ne garder que les colonnes existantes
numerical_cols = [col for col in numerical_cols if col in df.columns]

for col in numerical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

# Colonnes binaires (tinyint)
binary_cols = [
    'has_group_by', 'has_order_by', 'has_having', 'has_union',
    'has_distinct', 'has_limit', 'has_star_select', 'has_wildcard_like',
    'has_case_when', 'is_select_query', 'is_insert_query', 'is_update_query',
    'is_delete_query', 'is_peak_hour', 'is_weekend', 'is_business_hours',
    'connections_high'
]

binary_cols = [col for col in binary_cols if col in df.columns]

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)

# Colonne cible
df['is_slow'] = df['is_slow'].fillna(0).astype(int)

# Vérifier la distribution de la variable cible
print(f"📊 Distribution de la variable cible 'is_slow':")
print(df['is_slow'].value_counts())
print(f"Ratio: {df['is_slow'].mean():.2%}")

# ===============================
# NORMALISATION DES FEATURES NUMÉRIQUES
# ===============================
print("\n📐 Normalisation des features numériques...")
scaler = StandardScaler()
if numerical_cols:
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print(f"Features numériques normalisées: {len(numerical_cols)}")

# ===============================
# GÉNÉRATION DES EMBEDDINGS SQL
# ===============================
print("\n⏳ Génération des embeddings SQL...")

# Initialiser le modèle d'embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Préparer les textes SQL
df['sql_text'] = df['sql_text'].fillna("").astype(str)

# Vérifier la longueur des requêtes SQL
df['sql_text_length'] = df['sql_text'].str.len()
print(f"📏 Longueur moyenne des requêtes SQL: {df['sql_text_length'].mean():.0f} caractères")
print(f"📏 Longueur max: {df['sql_text_length'].max():.0f}, min: {df['sql_text_length'].min():.0f}")

# Générer les embeddings
sql_texts = df['sql_text'].tolist()

print(f"Génération des embeddings pour {len(sql_texts)} requêtes SQL...")
sql_embeddings = model.encode(
    sql_texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True  # Normaliser les embeddings pour meilleures performances
)

sql_embeddings = np.array(sql_embeddings)
print(f"✅ Embeddings SQL shape: {sql_embeddings.shape}")
print(f"📐 Dimension des embeddings: {sql_embeddings.shape[1]}")

# ===============================
# CRÉATION DU DATASET COMPLET
# ===============================
print("\n🏗️  Construction du dataset final...")

# Créer un DataFrame avec toutes les features structurées
feature_columns = numerical_cols + binary_cols + ['is_slow']
df_features = df[feature_columns].copy()

# Ajouter les embeddings comme nouvelles colonnes
for i in range(sql_embeddings.shape[1]):
    df_features[f'sql_embedding_{i:03d}'] = sql_embeddings[:, i]

# Ajouter les métadonnées pour référence
df_features['sql_text'] = df_original['sql_text']
df_features['sql_hash'] = df_original['sql_text'].apply(lambda x: hash(str(x)) % 1000000)

# Réorganiser les colonnes
embedding_cols = [f'sql_embedding_{i:03d}' for i in range(sql_embeddings.shape[1])]
final_columns = feature_columns + embedding_cols + ['sql_text', 'sql_hash']
df_final = df_features[final_columns]

# ===============================
# ANALYSE DU DATASET FINAL
# ===============================
print("\n📊 Analyse du dataset final:")
print(f"Dimensions: {df_final.shape}")
print(f"Nombre de features: {len(df_final.columns) - 3}")  # -3 pour sql_text, sql_hash, is_slow
print(f"Nombre d'embeddings: {sql_embeddings.shape[1]}")
print(f"Taille mémoire approximative: {df_final.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

# Vérifier les valeurs manquantes
missing_values = df_final.isnull().sum().sum()
if missing_values > 0:
    print(f"⚠️  Valeurs manquantes détectées: {missing_values}")
    # Remplir les dernières valeurs manquantes
    df_final = df_final.fillna(0)
else:
    print("✅ Aucune valeur manquante")

# ===============================
# SAUVEGARDE
# ===============================
output_path = "../data/exports/dataset_embedding_training.csv"
print(f"\n💾 Sauvegarde du dataset vers: {output_path}")
df_final.to_csv(output_path, index=False)

# Sauvegarder aussi une version réduite (sans texte SQL) pour l'entraînement
df_training = df_final.drop(['sql_text', 'sql_hash'], axis=1)
training_path = "../data/exports/dataset_embedding_training_clean.csv"
df_training.to_csv(training_path, index=False)

# Sauvegarder les informations du scaler
import joblib
scaler_path = "../data/exports/scaler.pkl"
joblib.dump(scaler, scaler_path)

print("\n✅ Dataset prêt pour l'entraînement!")
print("=" * 50)
print("📁 Fichiers générés:")
print(f"1. Dataset complet: {output_path}")
print(f"2. Dataset pour entraînement (sans texte): {training_path}")
print(f"3. Scaler sauvegardé: {scaler_path}")
print("=" * 50)
print(f"🎯 Variable cible: 'is_slow'")
print(f"📈 Ratio de classes: {df_final['is_slow'].mean():.2%}")
print(f"🔢 Nombre total de features: {df_training.shape[1]}")
print(f"📊 Nombre d'échantillons: {df_training.shape[0]}")