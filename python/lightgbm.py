import pandas as pd
import mysql.connector

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3308,
    'user': 'apbd_user',
    'password': 'apbd_pass',
    'database': 'pos'
}

# Load
conn = mysql.connector.connect(**DB_CONFIG)
df = pd.read_sql("SELECT * FROM dataset_ml", conn)
conn.close()

numerical_features = [
    'num_joins','num_where','has_order_by','has_group_by',
    'query_length','rows_examined','using_filesort',
    'using_temporary','index_count','buffer_pool_hit_ratio',
    'connections_count','hour_of_day','day_of_week'
]

categorical_features = ['access_type']

targets = ['is_slow', 'execution_time']

# Cleaning
df = df.dropna(subset=['is_slow'])
df[numerical_features] = df[numerical_features].fillna(0)
df[categorical_features] = df[categorical_features].fillna("UNKNOWN")

# LightGBM dataset
df_lgbm = df[numerical_features + categorical_features + targets]

df_lgbm.to_csv("../data/exports/dataset_lightgbm.csv", index=False)

print("✅ LightGBM dataset:", df_lgbm.shape)
