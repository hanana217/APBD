from db import get_conn
import csv

conn = get_conn()
cur = conn.cursor(dictionary=True)

# Slow queries
cur.execute("""
SELECT *
FROM performance_schema.events_statements_summary_by_digest
ORDER BY AVG_TIMER_WAIT DESC
LIMIT 100
""")

rows = cur.fetchall()

with open("data/exports/perf_digest.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Metrics exported")
