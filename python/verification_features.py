# verification_features.py
import mysql.connector
import pandas as pd

DB_CONFIG = {
    'host': 'localhost',     
    'port': 3308,            
    'user': 'apbd_user',
    'password': 'apbd_pass',
    'database': 'pos'
}


conn = mysql.connector.connect(**DB_CONFIG)
df = pd.read_sql("SELECT * FROM dataset_ml LIMIT 10", conn)
conn.close()

print("=" * 60)
print("VÉRIFICATION DES FEATURES EXTRACTES")
print("=" * 60)

# 1. Features de REQUÊTE
print("\n1. 📝 FEATURES DE REQUÊTE :")
req_features = ['num_joins', 'num_where', 'has_order_by', 'has_group_by', 'query_length']
for feat in req_features:
    if feat in df.columns:
        print(f"   ✅ {feat}: présent (ex: {df[feat].iloc[0]})")
    else:
        print(f"   ❌ {feat}: absent")

# 2. Features de STRUCTURE
print("\n2. 📊 FEATURES DE STRUCTURE (EXPLAIN):")
explain_features = ['access_type', 'key_used', 'using_filesort', 'using_temporary', 'index_count']
for feat in explain_features:
    if feat in df.columns:
        non_null = df[feat].notna().sum()
        print(f"   ✅ {feat}: présent ({non_null} valeurs non-null)")
    else:
        print(f"   ❌ {feat}: absent")

# 3. Features de CONTEXTE
print("\n3. 🖥️ FEATURES DE CONTEXTE SERVEUR :")
context_features = ['buffer_pool_hit_ratio', 'connections_count']
for feat in context_features:
    if feat in df.columns:
        val = df[feat].iloc[0]
        print(f"   ✅ {feat}: présent (ex: {val})")
    else:
        print(f"   ❌ {feat}: absent")

# 4. Target variables
print("\n4. 🎯 VARIABLES CIBLE :")
targets = ['is_slow', 'execution_time']
for target in targets:
    if target in df.columns:
        print(f"   ✅ {target}: présent")
        print(f"      Distribution is_slow: {df['is_slow'].value_counts().to_dict()}")
    else:
        print(f"   ❌ {target}: absent")

print("\n" + "=" * 60)
print(f"Total colonnes: {len(df.columns)}")
print("Colonnes disponibles:", list(df.columns))