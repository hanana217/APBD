import pandas as pd
from sqlalchemy import create_engine, text

# Connexion SQLAlchemy
engine = create_engine(
    "mysql+pymysql://root:@localhost/pos",
    echo=False
)

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
        return "EXPLAIN_FAILED"

# Charger les requêtes
df = pd.read_sql(
    "SELECT id, sql_text FROM dataset_ml",
    engine
)

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
