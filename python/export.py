import pandas as pd
import mysql.connector

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3307,
    'user': 'apbd_user',
    'password': 'apbd_pass',
    'database': 'pos'
}

conn = mysql.connector.connect(DB_CONFIG)

df = pd.read_sql("SELECT * FROM dataset_ml", conn)
conn.close()
