from mysql_utils import get_connection

print("Cleaning up test indexes...")

conn = get_connection()
cursor = conn.cursor()

try:
    cursor.execute("SHOW INDEX FROM orders")
    indexes = cursor.fetchall()
    
    dropped_count = 0
    for idx in indexes:
        if idx[2].startswith('idx_orders'):
            try:
                cursor.execute(f"DROP INDEX {idx[2]} ON orders")
                print(f"Dropped: {idx[2]}")
                dropped_count += 1
            except Exception as e:
                print(f"Failed to drop {idx[2]}: {e}")
    
    conn.commit()
    print(f"\nCleaned up {dropped_count} indexes.")
    
except Exception as e:
    print(f"Error during cleanup: {e}")
finally:
    cursor.close()
    conn.close()