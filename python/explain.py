import pandas as pd
from sqlalchemy import create_engine, text

# Use consistent configuration
MYSQL_CONFIG = {
    'host': '127.0.0.1',
    'port': 3308,         
    'user': 'apbd_user',
    'password': 'apbd_pass',
    'database': 'pos'
}


# Create engine with proper connection string
connection_string = f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
engine = create_engine(connection_string, echo=False)

# Test connection first
def test_connection():
    try:
        with engine.connect() as conn:
            print("✅ Connected to MySQL successfully!")
            return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"   Trying to connect to: {connection_string}")
        return False

if not test_connection():
    # Try alternative ports
    for port in [3306, 3307, 3308, 3309]:
        print(f"Trying port {port}...")
        connection_string = f"mysql+pymysql://root:@localhost:{port}/pos"
        engine = create_engine(connection_string, echo=False)
        try:
            with engine.connect() as conn:
                print(f"✅ Connected on port {port}!")
                break
        except:
            continue
    else:
        print("❌ Could not connect to MySQL on any common port")
        print("Please ensure MySQL is running and accessible")
        exit(1)

# Rest of your code...
# Fonction EXPLAIN → texte
def explain_to_text(sql):
    try:
        with engine.connect() as conn:
            result = conn.execute(text("EXPLAIN " + sql))
            rows = result.mappings().all()

        tokens = []
        for r in rows:
            tokens.append(
                f"TABLE={r['table']} "
                f"ACCESS={r['type']} "
                f"KEY={r['key']} "
                f"ROWS={r['rows']} "
                f"EXTRA={r['Extra']}"
            )
        return " | ".join(tokens)

    except Exception as e:
        return f"EXPLAIN_FAILED: {str(e)}"

# Charger les requêtes
try:
    df = pd.read_sql("SELECT id, sql_text FROM dataset_ml", engine)
    print(f"Loaded {len(df)} queries from database")
    
    # Générer explain_text
    df["explain_text"] = df["sql_text"].apply(explain_to_text)
    
    # Mise à jour en base
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text("UPDATE dataset_ml SET explain_text=:exp WHERE id=:id"),
                {"exp": row["explain_text"], "id": int(row["id"])}
            )
    
    print("✅ explain_text généré et sauvegardé correctement")
    
except Exception as e:
    print(f"❌ Error: {e}")