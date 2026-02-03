from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
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

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         categorical_features)
    ]
)


X = df[numerical_features + categorical_features]
X_ohe = preprocessor.fit_transform(X)

feature_names = (
    numerical_features +
    list(preprocessor.named_transformers_['cat']
         .get_feature_names_out(categorical_features))
)

df_xgb = pd.DataFrame(X_ohe, columns=feature_names)
df_xgb['is_slow'] = df['is_slow'].values
df_xgb['execution_time'] = df['execution_time'].values

df_xgb.to_csv("../data/exports/dataset_xgboost.csv", index=False)

print("✅ XGBoost dataset:", df_xgb.shape)
