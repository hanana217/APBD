import random
import time
from db import get_conn

queries = [
    "SELECT * FROM orders WHERE price > 100",
    "SELECT * FROM products ORDER BY price DESC",
    """
    SELECT *
    FROM orders o
    JOIN clients c ON o.client_id = c.id
    WHERE c.lastname LIKE '%a%'
    """
]

conn = get_conn()
cur = conn.cursor()

def peak_hours():
    for _ in range(200):
        cur.execute(random.choice(queries))
        cur.fetchall()

def low_traffic():
    for _ in range(20):
        cur.execute(random.choice(queries))
        cur.fetchall()

while True:
    hour = time.localtime().tm_hour
    if 9 <= hour <= 18:
        peak_hours()
    else:
        low_traffic()
    time.sleep(2)
