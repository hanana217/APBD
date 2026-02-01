import mysql.connector
import re
from datetime import datetime
import random

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3308,         
    'user': 'apbd_user',
    'password': 'apbd_pass',
    'database': 'pos'
}


LOG_FILES = [
    '..\docker\mysql\logs\slow.log'
]


SLOW_THRESHOLD = 1.0

def generate_fast_queries(cursor, limit=250, max_attempts=4000):
    """
    Génère des requêtes FAST (< SLOW_THRESHOLD)
    avec protection contre les boucles infinies
    """
    fast_queries = []
    tested = set()
    attempts = 0

    tables = [
        "clients", "products", "orders", "offers",
        "promotions", "cart", "admin", "wilayas"
    ]

    conditions = [
        "id > 0",
        "id < 1000",
        "id BETWEEN 10 AND 500",
        "id IS NOT NULL"
    ]

    limits = [5, 10, 20, 50, 100 , 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 5500]

    while len(fast_queries) < limit and attempts < max_attempts:
        attempts += 1

        table = random.choice(tables)
        condition = random.choice(conditions)
        limit_val = random.choice(limits)

        sql = f"SELECT * FROM {table} WHERE {condition} LIMIT 3000"

        if sql in tested:
            continue
        tested.add(sql)

        try:
            start = datetime.now()
            cursor.execute(sql)
            cursor.fetchall()
            exec_time = (datetime.now() - start).total_seconds()

            if exec_time < SLOW_THRESHOLD:
                fast_queries.append({
                    'sql_text': sql,
                    'execution_time': exec_time,
                    'rows_examined': limit_val,
                    'is_slow': 0
                })

        except:
            continue

    print(f"ℹ️ FAST generation: {len(fast_queries)} requêtes trouvées "
          f"en {attempts} essais")

    return fast_queries

def parse_slow_log(file_path):
    """
    Parser un fichier slow query log MySQL
    Retourne une liste de dictionnaires avec les infos des requêtes
    """
    queries = []
    current_query = {}
    sql_lines = []
    
    print(f"\n📖 Lecture de {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"⚠️ Fichier {file_path} introuvable")
        return []
    except Exception as e:
        print(f"❌ Erreur lecture {file_path}: {e}")
        return []
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
        
        # Ligne de temps
        if line.startswith('# Time:'):
            # Sauvegarder la requête précédente
            if sql_lines:
                current_query['sql_text'] = ' '.join(sql_lines).strip()
                if current_query.get('sql_text'):
                    queries.append(current_query.copy())
                sql_lines = []
                current_query = {}
            
            # Parser le timestamp
            time_match = re.search(r'# Time:\s*(.+)', line)
            if time_match:
                time_str = time_match.group(1)
                try:
                    timestamp = datetime.fromisoformat(time_str.replace('Z', ''))
                    current_query['timestamp'] = timestamp
                except:
                    current_query['timestamp'] = datetime.now()
        
        # Ligne avec Query_time, Lock_time, Rows
        elif line.startswith('# Query_time:'):
            # Exemple: # Query_time: 5.123  Lock_time: 0.001  Rows_sent: 10  Rows_examined: 1000
            print(f"Ligne Query_time: {line}")
            query_time = re.search(r'Query_time:\s*([\d\.]+)', line)
            if query_time:
                current_query['execution_time'] = float(query_time.group(1))
            
            rows_examined = re.search(r'Rows_examined:\s*(\d+)', line)
            if rows_examined:
                current_query['rows_examined'] = int(rows_examined.group(1))
        
        # Ligne SET timestamp (ignorer)
        elif line.startswith('SET timestamp='):
            continue
        
        # Ligne use database
        elif line.lower().startswith('use '):
            continue
        
        # Commentaire (ignorer)
        elif line.startswith('#'):
            continue
        
        # Ligne SQL (tout le reste)
        else:
            sql_lines.append(line)
    
    # Sauvegarder la dernière requête
    if sql_lines:
        current_query['sql_text'] = ' '.join(sql_lines).strip()
        if current_query.get('timestamp') and current_query.get('execution_time') is not None:
            queries.append(current_query.copy())
    print(f"✅ {len(queries)} requêtes extraites de {file_path}")
    return queries\
    


# ============================================
# FONCTION 2 : Extraire Features SQL
# ============================================
def extract_sql_features(sql_text):
    """
    Extraire les features depuis le texte SQL
    """
    if not sql_text:
        return {
            'num_joins': 0,
            'num_where': 0,
            'has_order_by': 0,
            'has_group_by': 0,
            'query_length': 0
        }
    
    sql_upper = sql_text.upper()
    
    features = {
        'num_joins': len(re.findall(r'\bJOIN\b', sql_upper)),
        'num_where': len(re.findall(r'\bWHERE\b', sql_upper)),
        'has_order_by': 1 if 'ORDER BY' in sql_upper else 0,
        'has_group_by': 1 if 'GROUP BY' in sql_upper else 0,
        'query_length': len(sql_text)
    }
    
    return features

# ============================================
# FONCTION 3 : Extraire Features Temporelles
# ============================================
def extract_temporal_features(timestamp):
    """
    Extraire hour_of_day et day_of_week
    """
    if not timestamp:
        timestamp = datetime.now()
    
    return {
        'hour_of_day': timestamp.hour,
        'day_of_week': timestamp.weekday()  # 0=Lundi, 6=Dimanche
    }

# ============================================
# FONCTION 4 : Obtenir EXPLAIN Features
# ============================================
def get_explain_features(conn, sql_text):
    try:
        sql_clean = sql_text.strip().rstrip(';')
        with conn.cursor(buffered=True) as cur:
            cur.execute(f"EXPLAIN {sql_clean}")
            results = cur.fetchall()
        if not results:
            return default_explain()
        first = results[0]
        possible_keys = first[4]
        index_count = len(possible_keys.split(',')) if possible_keys else 0
        return {
            'access_type': first[3],
            'key_used': first[5],
            'index_count': index_count,
            'using_filesort': 1 if ('filesort' in str(first[9]).lower()) else 0,
            'using_temporary': 1 if ('temporary' in str(first[9]).lower()) else 0
        }
    except Exception:
        return default_explain()



def default_explain():
    return {
        'access_type': None,
        'key_used': None,
        'index_count': 0,
        'using_filesort': 0,
        'using_temporary': 0
    }



# ============================================
# FONCTION 5 : Obtenir Métriques Serveur
# ============================================
def get_server_metrics(cursor):
    """
    Obtenir cpu_usage, buffer_pool_hit_ratio, connections_count
    """
    metrics = {
        'cpu_usage': None,
        'buffer_pool_hit_ratio': None,
        'connections_count': None
    }
    
    try:
        # Connections count
        cursor.execute("SHOW STATUS LIKE 'Threads_connected'")
        result = cursor.fetchone()
        if result:
            metrics['connections_count'] = int(result[1])
        
        # Buffer pool hit ratio
        cursor.execute("SHOW STATUS LIKE 'Innodb_buffer_pool_read_requests'")
        read_requests = cursor.fetchone()
        cursor.execute("SHOW STATUS LIKE 'Innodb_buffer_pool_reads'")
        reads = cursor.fetchone()
        
        if read_requests and reads:
            total_requests = int(read_requests[1])
            disk_reads = int(reads[1])
            if total_requests > 0:
                metrics['buffer_pool_hit_ratio'] = round(1 - (disk_reads / total_requests), 4)
        
        # CPU usage (non disponible directement dans MySQL, on met NULL)
        metrics['cpu_usage'] = None
        
    except Exception as e:
        print(f"⚠️ Erreur métriques serveur: {e}")
    
    return metrics

# ============================================
# FONCTION 6 : Insérer dans dataset_ml
# ============================================
def insert_into_dataset_ml(cursor, query_data, server_metrics):

    sql = """
    INSERT INTO dataset_ml (
        sql_text,
        num_joins, num_where, has_order_by, has_group_by, query_length,
        rows_examined, access_type, key_used, using_filesort, using_temporary,
        cpu_usage, buffer_pool_hit_ratio, connections_count,
        hour_of_day, day_of_week, execution_time, is_slow,
        index_count
    ) VALUES (
        %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s,
        %s
    )
    """


    # Vérifier doublon
    cursor.execute(
        "SELECT 1 FROM dataset_ml WHERE sql_text = %s LIMIT 1",
        (query_data['sql_text'],)
    )
    result = cursor.fetchone()

    if result:
        return False

    values = (
        query_data.get('sql_text', ''),
        query_data.get('num_joins', 0),
        query_data.get('num_where', 0),
        query_data.get('has_order_by', 0),
        query_data.get('has_group_by', 0),
        query_data.get('query_length', 0),
        query_data.get('rows_examined', 0),
        query_data.get('access_type'),
        query_data.get('key_used'),
        query_data.get('using_filesort', 0),
        query_data.get('using_temporary', 0),
        server_metrics.get('cpu_usage'),
        server_metrics.get('buffer_pool_hit_ratio'),
        server_metrics.get('connections_count'),
        query_data.get('hour_of_day', 0),
        query_data.get('day_of_week', 0),
        query_data.get('execution_time', 0),
        query_data.get('is_slow', 0),
        query_data.get('index_count', 0)
    )


    cursor.execute(sql, values)
    return True


## ============================================
# FONCTION PRINCIPALE MODIFIÉE
# ============================================
def main():
    print("IMPORT DES SLOW QUERIES DANS dataset_ml")
    all_queries = []
    for log_file in LOG_FILES:
        queries = parse_slow_log(log_file)
        all_queries.extend(queries)
    
    if not all_queries:
        print("\nAucune requête trouvée dans les fichiers de logs")
        return
    
    print(f"\nTotal de requêtes extraites: {len(all_queries)}")
    
    # 2. Connexion à MySQL
    print(f"\n🔌 Connexion à MySQL...")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor_read = conn.cursor(buffered=True)   # Pour EXPLAIN et SELECT
        cursor_write = conn.cursor(buffered=True)  # Pour INSERT
        cursor_fast  = conn.cursor(buffered=True)  # Pour FAST queries


        print("Connecté à MySQL")
    except Exception as e:
        print(f"Erreur de connexion: {e}")
        return
    
    # 3. Obtenir les métriques serveur (une seule fois)
    print(f"\nRécupération des métriques serveur...")
    server_metrics = get_server_metrics(cursor_read)
    print(f"Métriques récupérées:")
    print(f"   • Connections: {server_metrics['connections_count']}")
    print(f"   • Buffer pool hit ratio: {server_metrics['buffer_pool_hit_ratio']}")
    
    # 4. Traiter chaque requête
    print(f"\nTraitement des requêtes lentes...")
    inserted_count = 0
    error_count = 0
    seen_queries = set()  # pour éviter les doublons

    for i, query in enumerate(all_queries, 1):
        try:
            sql_text = query.get('sql_text', '').strip()
            execution_time = query.get('execution_time', 0)
            
            # Filtrer : uniquement les requêtes lentes
            if execution_time < SLOW_THRESHOLD:
                continue
            
            # Filtrer : éviter les doublons
            if sql_text in seen_queries:
                continue
            seen_queries.add(sql_text)
            
            # Features SQL
            sql_features = extract_sql_features(sql_text)
            
            # Features temporelles
            temporal_features = extract_temporal_features(query.get('timestamp'))
            
            # Features EXPLAIN (optionnel)
            explain_features = get_explain_features(cursor_read, sql_text)
            
            # Combiner toutes les features
            query_data = {
                'sql_text': sql_text,
                **sql_features,
                **temporal_features,
                **explain_features,
                'rows_examined': query.get('rows_examined', 0),
                'execution_time': execution_time,
                'is_slow': 1
            }

            
            # Insérer dans la base
            insert_into_dataset_ml(cursor_write, query_data, server_metrics)
            conn.commit()
            
            inserted_count += 1

        except Exception as e:
            error_count += 1
            print(f"   ⚠️ Erreur requête {i}: {e}")
            continue
        # ============================================
    # AJOUT DES REQUÊTES RAPIDES (FAST)
    # ============================================
    print("\nAjout des requêtes FAST (< 1s)...")

    fast_queries = generate_fast_queries(cursor_fast, limit=250)
    fast_inserted = 0

    for query in fast_queries:
        sql_text = query['sql_text']

        # utiliser un curseur temporaire pour vérifier les doublons
        with conn.cursor(buffered=True) as cur:
            cur.execute(
                "SELECT 1 FROM dataset_ml WHERE sql_text = %s LIMIT 1",
                (sql_text,)
            )
            if cur.fetchone():
                continue

        sql_features = extract_sql_features(sql_text)
        temporal_features = extract_temporal_features(datetime.now())

        query_data = {
            'sql_text': sql_text,
            **sql_features,
            **temporal_features,
            'rows_examined': query.get('rows_examined', 0),
            'execution_time': query['execution_time'],
            'is_slow': 0,
            'index_count': 0
        }

        insert_into_dataset_ml(cursor_write, query_data, server_metrics)
        conn.commit()

        fast_inserted += 1


    print(f"✅ Requêtes FAST insérées: {fast_inserted}")

    # 5. Fermer la connexion
    cursor_read.close()
    cursor_write.close()
    cursor_fast.close()
    conn.close()

    # 6. Résumé
    print("\nIMPORT TERMINÉ")
    print("=" * 80)
    print(f"""
📊 Résumé:
   • Requêtes lentes trouvées: {len(seen_queries)}
   • Requêtes insérées: {inserted_count}
   • Erreurs: {error_count}
   • Seuil is_slow: {SLOW_THRESHOLD}s
""")

if __name__ == "__main__":
    main()