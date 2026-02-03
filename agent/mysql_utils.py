# mysql_utils.py - Version CORRIGÉE
import mysql.connector
import time
import sys
import os

# Détermine le chemin du projet
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docker_config_path = os.path.join(project_root, 'docker', 'rl_agent', 'config_docker.py')

# Import automatique selon le mode
try:
    # Essaie d'abord d'importer config_docker depuis le dossier Docker
    import importlib.util
    
    if os.path.exists(docker_config_path):
        spec = importlib.util.spec_from_file_location("config_docker", docker_config_path)
        config_docker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_docker)
        MYSQL_CONFIG = config_docker.MYSQL_CONFIG
        print("🔗 Mode DOCKER activé (fichier trouvé)")
    else:
        raise ImportError("Fichier Docker non trouvé")
        
except ImportError:
    # Sinon, mode local
    try:
        from config import MYSQL_CONFIG
        print("💻 Mode LOCAL activé (config.py)")
    except ImportError:
        # Fallback: configuration par défaut
        print("⚠  Aucun config trouvé, utilisation des valeurs par défaut")
        MYSQL_CONFIG = {
            'host': 'localhost',
            'port': 3308,
            'user': 'apbd_user',
            'password': 'apbd_pass',
            'database': 'pos',
            'autocommit': True
        }

def get_connection():
    """Get MySQL connection with better error handling"""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        print(f"✅ Connecté à MySQL sur {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}")
        return conn
    except mysql.connector.Error as err:
        print(f"❌ ERREUR connexion MySQL: {err}")
        print(f"   Vérifie que Docker est lancé: docker compose up")
        print(f"   Configuration utilisée: {MYSQL_CONFIG}")
        sys.exit(1)
        
def fetch_all_results(cursor):
    """Helper function to fetch all results and clear the buffer"""
    try:
        results = cursor.fetchall()
        # Consume any remaining results
        while cursor.nextset():
            try:
                cursor.fetchall()
            except:
                pass
        return results
    except:
        return []

def measure_query_performance(cursor):
    queries = [
        """
        SELECT COUNT(*)
        FROM orders o
        JOIN clients c ON o.client_id = c.id
        WHERE c.wilaya = 16 AND o.price > 1000
        """,
        """
        SELECT *
        FROM orders
        WHERE client_id = 10
        ORDER BY orderdate DESC
        LIMIT 10
        """,
        """
        SELECT c.wilaya, AVG(o.price) as avg_price
        FROM orders o
        JOIN clients c ON o.client_id = c.id
        GROUP BY c.wilaya
        ORDER BY avg_price DESC
        LIMIT 5
        """,
        """
        SELECT p.category, COUNT(o.id) as order_count
        FROM orders o
        JOIN cart ct ON o.cart_id = ct.id
        JOIN products p ON ct.product_id = p.id
        WHERE o.orderdate > DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY p.category
        ORDER BY order_count DESC
        """
    ]
    
    total_time = 0
    for i, q in enumerate(queries):
        try:
            start = time.time()
            cursor.execute(q)
            fetch_all_results(cursor)  # Consume all results
            query_time = time.time() - start
            total_time += query_time
        except Exception as e:
            print(f"Query {i+1} failed: {e}")
            total_time += 1.0
    
    return total_time / len(queries)

def get_existing_indexes(cursor):
    try:
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.statistics
            WHERE table_schema = DATABASE()
            AND table_name = 'orders'
            AND index_name LIKE 'idx_orders_%'
        """)
        result = cursor.fetchone()
        cursor.fetchall()  # Clear any remaining results
        return result[0] if result else 0
    except Exception as e:
        print(f"Error getting existing indexes: {e}")
        return 0