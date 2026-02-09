"""
Enhanced MySQL utilities with binary matrix support
KEYWORD: ENHANCED_MYSQL_UTILS
"""

import mysql.connector
import time
import sys
import os
import numpy as np

# Import configuration
try:
    from config import MYSQL_CONFIG, SCHEMA_DEFINITION
    print("🔗 Enhanced config loaded")
except ImportError:
    print("⚠️ Using default config")
    MYSQL_CONFIG = {
        'host': 'localhost',
        'port': 3308,
        'user': 'apbd_user',
        'password': 'apbd_pass',
        'database': 'pos'
    }
    SCHEMA_DEFINITION = {}

def get_connection():
    """Get MySQL connection with enhanced error handling"""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        print(f"✅ Connecté à MySQL sur {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}")
        return conn
    except mysql.connector.Error as err:
        print(f"❌ ERREUR connexion MySQL: {err}")
        print(f"   Vérifie que Docker est lancé: docker compose up")
        sys.exit(1)

def fetch_all_results(cursor):
    """Helper function to fetch all results and clear the buffer"""
    try:
        results = cursor.fetchall()
        while cursor.nextset():
            try:
                cursor.fetchall()
            except:
                pass
        return results
    except:
        return []

def measure_query_performance(cursor):
    """Measure SELECT query performance with multiple query types"""
    queries = [
        """SELECT COUNT(*) FROM orders o JOIN clients c ON o.client_id = c.id 
           WHERE c.wilaya = 16 AND o.price > 1000""",
        """SELECT * FROM orders WHERE client_id = 10 
           ORDER BY orderdate DESC LIMIT 10""",
        """SELECT c.wilaya, AVG(o.price) as avg_price 
           FROM orders o JOIN clients c ON o.client_id = c.id 
           GROUP BY c.wilaya ORDER BY avg_price DESC LIMIT 5""",
        """SELECT p.category, COUNT(o.id) as order_count 
           FROM orders o JOIN cart ct ON o.cart_id = ct.id 
           JOIN products p ON ct.product_id = p.id 
           WHERE o.orderdate > DATE_SUB(NOW(), INTERVAL 30 DAY) 
           GROUP BY p.category ORDER BY order_count DESC"""
    ]
    
    total_time = 0
    for i, q in enumerate(queries):
        try:
            start = time.time()
            cursor.execute(q)
            fetch_all_results(cursor)
            total_time += time.time() - start
        except Exception as e:
            print(f"Query {i+1} failed: {e}")
            total_time += 1.0
    
    return total_time / len(queries)

def measure_insert_performance(cursor, table='orders'):
    """Measure INSERT performance to detect index impact"""
    try:
        # Create test table if it doesn't exist
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS perf_test_insert (
                id INT AUTO_INCREMENT PRIMARY KEY,
                data VARCHAR(255),
                value FLOAT,
                timestamp DATETIME,
                test_column1 VARCHAR(100),
                test_column2 INT,
                test_column3 DECIMAL(10,2)
            ) ENGINE=InnoDB
        """)
        
        # Measure batch INSERT time
        start_time = time.time()
        batch_size = 100
        
        for i in range(batch_size):
            cursor.execute(f"""
                INSERT INTO perf_test_insert 
                (data, value, timestamp, test_column1, test_column2, test_column3)
                VALUES ('test_data_{i}', {i * 1.5}, NOW(), 'category_{i%10}', {i%100}, {i * 0.1})
            """)
        
        cursor.execute("COMMIT")
        insert_time = (time.time() - start_time) / batch_size
        
        # Clean up
        cursor.execute("TRUNCATE TABLE perf_test_insert")
        
        return insert_time
    except Exception as e:
        print(f"[INSERT Measurement Error] {e}")
        return 0.01

def get_binary_index_matrix(cursor):
    """
    Create binary matrix representation of existing indexes
    Returns: numpy array where 1=indexed, 0=not indexed
    KEYWORD: BINARY_MATRIX
    """
    if not SCHEMA_DEFINITION:
        return np.array([])
    
    matrix = []
    
    try:
        for table, schema in SCHEMA_DEFINITION.items():
            # Get all non-primary indexes for this table
            cursor.execute(f"""
                SELECT COLUMN_NAME, INDEX_NAME
                FROM INFORMATION_SCHEMA.STATISTICS
                WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME = '{table}'
                AND INDEX_NAME != 'PRIMARY'
            """)
            
            existing_indexes = cursor.fetchall()
            # Create mapping of column to indexed status
            indexed_columns = {col: 0 for col in schema['columns']}
            
            for col, idx_name in existing_indexes:
                if col in indexed_columns:
                    indexed_columns[col] = 1
            
            # Add to matrix
            matrix.extend([indexed_columns[col] for col in schema['columns']])
        
        return np.array(matrix, dtype=np.float32)
    
    except Exception as e:
        print(f"[Binary Matrix Error] {e}")
        # Return all zeros as fallback
        total_cols = sum(len(schema['columns']) for schema in SCHEMA_DEFINITION.values())
        return np.zeros(total_cols, dtype=np.float32)

def get_existing_indexes(cursor):
    """Get count of existing test indexes"""
    try:
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.statistics
            WHERE table_schema = DATABASE()
            AND table_name = 'orders'
            AND index_name LIKE 'idx_orders_%'
        """)
        result = cursor.fetchone()
        return result[0] if result else 0
    except Exception as e:
        print(f"Error getting existing indexes: {e}")
        return 0

def get_index_usage_stats(cursor):
    """Get index usage statistics if available"""
    try:
        cursor.execute("""
            SELECT 
                TABLE_NAME,
                INDEX_NAME,
                COLUMN_NAME,
                CARDINALITY
            FROM INFORMATION_SCHEMA.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE()
            ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX
        """)
        return cursor.fetchall()
    except:
        return []

def get_table_statistics(cursor):
    """Get table size and row count statistics"""
    stats = {}
    try:
        cursor.execute("""
            SELECT 
                TABLE_NAME,
                TABLE_ROWS,
                DATA_LENGTH / 1024 / 1024 as SIZE_MB
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = DATABASE()
            AND TABLE_TYPE = 'BASE TABLE'
        """)
        
        for table_name, rows, size_mb in cursor.fetchall():
            stats[table_name] = {
                'rows': rows or 0,
                'size_mb': float(size_mb or 0)
            }
    except Exception as e:
        print(f"[Table Stats Error] {e}")
    
    return stats