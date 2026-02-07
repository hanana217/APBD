import mysql.connector
import re
from datetime import datetime
import math

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3308,
    'user': 'apbd_user',
    'password': 'apbd_pass',
    'database': 'pos'
}

LOG_FILES = [
    r'..\docker\mysql\logs\slow.log'
]

SLOW_THRESHOLD = 0.5
MAX_INDEXES = 5

# ======================================================
# PARSE SLOW LOG
# ======================================================
def parse_slow_log(file_path):
    queries = []
    current = {}
    sql_lines = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()

            if line.startswith('# Time:'):
                if sql_lines:
                    current['sql_text'] = ' '.join(sql_lines)
                    queries.append(current)
                    current = {}
                    sql_lines = []

                try:
                    t = line.split('# Time:')[1].strip().replace('Z', '')
                    current['timestamp'] = datetime.fromisoformat(t)
                except:
                    current['timestamp'] = datetime.now()

            elif line.startswith('# Query_time:'):
                m = re.search(r'Query_time:\s*([\d\.]+)', line)
                if m:
                    current['execution_time'] = float(m.group(1))

            elif line.startswith('SET timestamp') or line.startswith('#'):
                continue

            else:
                sql_lines.append(line)

    if sql_lines:
        current['sql_text'] = ' '.join(sql_lines)
        queries.append(current)

    print(f"✅ {len(queries)} requêtes extraites")
    return queries


# ======================================================
# SQL FEATURES
# ======================================================
def extract_sql_features(sql):
    s = sql.upper()

    num_joins = len(re.findall(r'\bJOIN\b', s))
    num_where = len(re.findall(r'\bWHERE\b', s))
    num_tables = len(re.findall(r'\bFROM\b|\bJOIN\b', s))
    num_predicates = len(re.findall(r'\bAND\b|\bOR\b', s))

    return {
        'query_length': len(sql),
        'query_length_log': math.log(len(sql) + 1),

        'num_joins': num_joins,
        'num_where': num_where,
        'num_tables': num_tables,
        'num_predicates': num_predicates,

        'has_group_by': int('GROUP BY' in s),
        'has_order_by': int('ORDER BY' in s),
        'has_having': int('HAVING' in s),
        'has_union': int('UNION' in s),
        'has_distinct': int('DISTINCT' in s),
        'has_limit': int('LIMIT' in s),
        'has_star_select': int('SELECT *' in s),

        'num_subqueries': len(re.findall(r'\(SELECT', s)),
        'num_aggregates': len(re.findall(r'\bSUM\b|\bCOUNT\b|\bAVG\b|\bMIN\b|\bMAX\b', s)),
        'num_functions': len(re.findall(r'\w+\(', s)),

        'max_in_list_size': 0,
        'has_wildcard_like': int('LIKE \'%\'' in s),
        'has_case_when': int('CASE WHEN' in s),

        'is_select_query': int(s.startswith('SELECT')),
        'is_insert_query': int(s.startswith('INSERT')),
        'is_update_query': int(s.startswith('UPDATE')),
        'is_delete_query': int(s.startswith('DELETE')),
    }


# ======================================================
# TEMPORAL FEATURES
# ======================================================
def extract_temporal_features(ts):
    hour = ts.hour
    day = ts.weekday()

    return {
        'hour_of_day': hour,
        'day_of_week': day,
        'is_peak_hour': int(9 <= hour <= 17),
        'is_weekend': int(day >= 5),
        'is_business_hours': int(9 <= hour <= 17 and day < 5)
    }


# ======================================================
# SERVER METRICS
# ======================================================
def get_server_metrics(cursor):
    cursor.execute("SHOW STATUS LIKE 'Threads_connected'")
    connections = int(cursor.fetchone()[1])

    cursor.execute("SHOW STATUS LIKE 'Innodb_buffer_pool_read_requests'")
    rr = int(cursor.fetchone()[1])
    cursor.execute("SHOW STATUS LIKE 'Innodb_buffer_pool_reads'")
    r = int(cursor.fetchone()[1])

    hit_ratio = round(1 - (r / rr), 4) if rr > 0 else 0.5

    return {
        'connections_count': connections,
        'connections_high': int(connections > 50),
        'buffer_pool_hit_ratio': hit_ratio
    }


# ======================================================
# INSERT DATASET_FINAL
# ======================================================
def insert_dataset_final(cursor, data):
    sql = """
    INSERT IGNORE INTO dataset_final (
        query_length, query_length_log,
        num_joins, num_where,
        has_group_by, has_order_by, has_having,
        has_union, has_distinct, has_limit, has_star_select,
        num_subqueries, num_predicates, num_aggregates,
        num_functions, num_tables,
        max_in_list_size,
        has_wildcard_like, has_case_when,
        is_select_query, is_insert_query,
        is_update_query, is_delete_query,
        hour_of_day, day_of_week,
        is_peak_hour, is_weekend, is_business_hours,
        connections_count, connections_high,
        buffer_pool_hit_ratio,
        estimated_index_count, estimated_table_size_mb,
        joins_per_table, predicates_per_where,
        complexity_density,
        is_slow,
        sql_text
    ) VALUES (
        %(query_length)s, %(query_length_log)s,
        %(num_joins)s, %(num_where)s,
        %(has_group_by)s, %(has_order_by)s, %(has_having)s,
        %(has_union)s, %(has_distinct)s, %(has_limit)s, %(has_star_select)s,
        %(num_subqueries)s, %(num_predicates)s, %(num_aggregates)s,
        %(num_functions)s, %(num_tables)s,
        %(max_in_list_size)s,
        %(has_wildcard_like)s, %(has_case_when)s,
        %(is_select_query)s, %(is_insert_query)s,
        %(is_update_query)s, %(is_delete_query)s,
        %(hour_of_day)s, %(day_of_week)s,
        %(is_peak_hour)s, %(is_weekend)s, %(is_business_hours)s,
        %(connections_count)s, %(connections_high)s,
        %(buffer_pool_hit_ratio)s,
        %(estimated_index_count)s, %(estimated_table_size_mb)s,
        %(joins_per_table)s, %(predicates_per_where)s,
        %(complexity_density)s,
        %(is_slow)s,
        %(sql_text)s
    )
    """
    cursor.execute(sql, data)



# ======================================================
# MAIN
# ======================================================
def main():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(buffered=True)

    server_metrics = get_server_metrics(cursor)

    total = 0
    slow = 0

    for log in LOG_FILES:
        for q in parse_slow_log(log):
            sql = q.get('sql_text', '')
            exec_time = q.get('execution_time', 0)
            ts = q.get('timestamp', datetime.now())

            sql_f = extract_sql_features(sql)
            time_f = extract_temporal_features(ts)

            num_tables = max(sql_f['num_tables'], 1)

            row = {
                **sql_f,
                **time_f,
                **server_metrics,

                'estimated_index_count': 0,
                'estimated_table_size_mb': 10,
                'joins_per_table': sql_f['num_joins'] / num_tables,
                'predicates_per_where': sql_f['num_predicates'] / max(sql_f['num_where'], 1),
                'complexity_density': sql_f['query_length'] / num_tables,

                'is_slow': int(exec_time > SLOW_THRESHOLD),
                'sql_text': sql.strip()
            }


            insert_dataset_final(cursor, row)
            total += 1
            slow += row['is_slow']

    conn.commit()
    cursor.close()
    conn.close()

    print(f"""
✅ IMPORT TERMINÉ
-----------------------------
Total requêtes : {total}
Requêtes lentes (>0.5s) : {slow}
Requêtes rapides : {total - slow}
""")

if __name__ == "__main__":
    main()
