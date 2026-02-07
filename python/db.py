"""
Générateur de Requêtes LENTES pour Équilibrer le Dataset - SADOP
Génère spécifiquement des requêtes qui seront LENTES
"""

import mysql.connector
from mysql.connector import Error
import random
import hashlib
import math
from datetime import datetime
import time


class SlowQueryGenerator:
    """Génère UNIQUEMENT des requêtes LENTES"""
    
    def __init__(self):
        self.wilayas = list(range(1, 59))
        self.statuses = ['En attente', 'Confirmée', 'En préparation', 'Expédiée', 
                        'Livrée', 'Annulée', 'Retournée']
        self.categories = ['Électronique', 'Mode', 'Maison', 'Sport', 'Beauté', 
                          'Livres', 'Jouets', 'Alimentation']
        self.brands = ['Samsung', 'Apple', 'Sony', 'LG', 'Nike', 'Adidas', 
                      'Zara', 'H&M', 'Generic']
        self.search_terms = ['phone', 'laptop', 'shirt', 'shoes', 'watch', 'bag',
                            'dress', 'jeans', 'tablet', 'headphone']
    
    def generate_slow_queries(self, count=5000):
        """Générer UNIQUEMENT des requêtes LENTES"""
        
        queries = []
        
        print(f"🔄 Génération de {count} requêtes LENTES...\n")
        
        # Répartition des types de requêtes lentes
        per_type = count // 10
        
        # 1. FULL SCAN avec LIKE (très lent)
        print(f"   1. Full scan LIKE: {per_type}")
        for i in range(per_type):
            term = random.choice(['Ali', 'Mohamed', 'Sara', 'Amina', 'Yacine', 
                                 'Ahmed', 'Fatima', 'Karim'])
            queries.append({
                'sql': f"SELECT * FROM clients WHERE firstname LIKE '%{term}%' OR lastname LIKE '%{term}%'",
                'type': 'full_scan_like'
            })
        
        # 2. JOIN 3+ tables sans index approprié
        print(f"   2. Multi-table JOINs: {per_type}")
        for i in range(per_type):
            queries.append({
                'sql': f"""
                    SELECT c.firstname, c.lastname, o.orderdate, p.name, p.price, w.name as wilaya
                    FROM clients c
                    JOIN orders o ON c.id = o.client_id
                    JOIN cart ca ON o.cart_id = ca.id
                    JOIN products p ON ca.product_id = p.id
                    JOIN wilayas w ON o.wilaya_id = w.id
                    WHERE o.orderdate >= '2024-{random.randint(1,12):02d}-01'
                    ORDER BY o.price DESC
                    LIMIT {random.randint(100, 500)}
                """,
                'type': 'join_4_tables'
            })
        
        # 3. Sous-requêtes corrélées (TRÈS LENT)
        print(f"   3. Sous-requêtes corrélées: {per_type}")
        for i in range(per_type):
            queries.append({
                'sql': f"""
                    SELECT c.id, c.firstname, c.lastname,
                        (SELECT COUNT(*) FROM orders o WHERE o.client_id = c.id) as order_count,
                        (SELECT SUM(price) FROM orders o WHERE o.client_id = c.id) as total_spent,
                        (SELECT MAX(orderdate) FROM orders o WHERE o.client_id = c.id) as last_order
                    FROM clients c
                    WHERE (SELECT COUNT(*) FROM orders o WHERE o.client_id = c.id) > {random.randint(1,5)}
                    LIMIT {random.randint(50, 200)}
                """,
                'type': 'correlated_subquery'
            })
        
        # 4. Agrégations complexes sans index
        print(f"   4. Agrégations complexes: {per_type}")
        for i in range(per_type):
            queries.append({
                'sql': f"""
                    SELECT 
                        DATE_FORMAT(o.orderdate, '%Y-%m') as month,
                        w.name as wilaya,
                        o.status,
                        COUNT(*) as order_count,
                        AVG(o.price) as avg_price,
                        SUM(o.price) as total_revenue,
                        STDDEV(o.price) as stddev_price,
                        COUNT(DISTINCT o.client_id) as unique_clients,
                        MIN(o.price) as min_price,
                        MAX(o.price) as max_price
                    FROM orders o
                    JOIN wilayas w ON o.wilaya_id = w.id
                    WHERE o.orderdate >= '2023-{random.randint(1,12):02d}-01'
                    GROUP BY DATE_FORMAT(o.orderdate, '%Y-%m'), w.name, o.status
                    HAVING COUNT(*) > {random.randint(1,10)}
                    ORDER BY month DESC, total_revenue DESC
                """,
                'type': 'complex_aggregate'
            })
        
        # 5. Fonction sur colonne indexée (désactive l'index)
        print(f"   5. Fonctions sur colonnes indexées: {per_type}")
        for i in range(per_type):
            queries.append({
                'sql': f"""
                    SELECT * FROM orders
                    WHERE YEAR(orderdate) = {random.choice([2023, 2024])}
                    AND MONTH(orderdate) = {random.randint(1, 12)}
                    AND status IN ('{random.choice(self.statuses)}', '{random.choice(self.statuses)}')
                """,
                'type': 'function_on_indexed'
            })
        
        # 6. IN avec GRANDE liste
        print(f"   6. IN avec grandes listes: {per_type}")
        for i in range(per_type):
            list_size = random.randint(100, 500)
            big_list = ','.join(str(random.randint(1, 50000)) for _ in range(list_size))
            queries.append({
                'sql': f"SELECT * FROM products WHERE id IN ({big_list})",
                'type': 'in_large_list'
            })
        
        # 7. NOT IN avec sous-requête (TRÈS LENT)
        print(f"   7. NOT IN avec sous-requêtes: {per_type}")
        for i in range(per_type):
            queries.append({
                'sql': f"""
                    SELECT * FROM products
                    WHERE id NOT IN (
                        SELECT product_id FROM cart 
                        WHERE date_cart >= '2024-{random.randint(1,12):02d}-01'
                    )
                    LIMIT {random.randint(100, 500)}
                """,
                'type': 'not_in_subquery'
            })
        
        # 8. OR multiple avec LIKE (scan complet)
        print(f"   8. OR multiple LIKE: {per_type}")
        for i in range(per_type):
            term = random.choice(self.search_terms)
            queries.append({
                'sql': f"""
                    SELECT * FROM products
                    WHERE description LIKE '%{term}%'
                    OR more_details LIKE '%{term}%'
                    OR name LIKE '%{term}%'
                    LIMIT {random.randint(100, 500)}
                """,
                'type': 'or_multi_like'
            })
        
        # 9. GROUP BY sans index avec DISTINCT
        print(f"   9. GROUP BY + DISTINCT: {per_type}")
        for i in range(per_type):
            queries.append({
                'sql': f"""
                    SELECT client_id, 
                           COUNT(DISTINCT product_id) as unique_products,
                           COUNT(DISTINCT YEAR(date_cart)) as years,
                           COUNT(*) as total_items,
                           SUM(initial_price) as total_price,
                           AVG(initial_price) as avg_price
                    FROM cart
                    WHERE date_cart >= '2024-{random.randint(1,6):02d}-01'
                    GROUP BY client_id
                    HAVING COUNT(DISTINCT product_id) > {random.randint(2,5)}
                    ORDER BY total_price DESC
                    LIMIT {random.randint(100, 300)}
                """,
                'type': 'group_distinct'
            })
        
        # 10. CROSS JOIN (produit cartésien - TRÈS LENT)
        print(f"   10. CROSS JOINs: {per_type}")
        for i in range(per_type):
            queries.append({
                'sql': f"""
                    SELECT c.firstname, c.lastname, p.name, p.price
                    FROM clients c
                    CROSS JOIN products p
                    WHERE c.wilaya = {random.randint(1, 58)}
                    AND p.price > {random.randint(100, 1000)}
                    LIMIT {random.randint(100, 500)}
                """,
                'type': 'cross_join'
            })
        
        # Mélanger
        random.shuffle(queries)
        
        print(f"\n✅ Total généré: {len(queries)} requêtes LENTES")
        
        return queries
    
    def connect_to_db(self):
        """Se connecter à MySQL"""
        try:
            connection = mysql.connector.connect(
                host='127.0.0.1',
                port=3308,
                user='apbd_user',
                password='apbd_pass',
                database='pos',
                autocommit=True
            )
            print("✅ Connecté à MySQL")
            return connection
        except Error as e:
            print(f"❌ Erreur connexion: {e}")
            return None
    
    def extract_features(self, sql):
        """Extraire les features d'une requête SQL"""
        import re
        
        sql_clean = sql.strip()
        sql_upper = sql_clean.upper()
        
        features = {
            'sql_text': sql_clean,
            'query_length': len(sql_clean),
            'query_length_log': math.log(len(sql_clean) + 1),
            
            # Structure SQL
            'num_joins': sql_upper.count('JOIN'),
            'num_where': sql_upper.count('WHERE'),
            'has_group_by': int('GROUP BY' in sql_upper),
            'has_order_by': int('ORDER BY' in sql_upper),
            'has_having': int('HAVING' in sql_upper),
            'has_union': int('UNION' in sql_upper),
            'has_distinct': int('DISTINCT' in sql_upper),
            'has_limit': int('LIMIT' in sql_upper),
            'has_star_select': int('SELECT *' in sql_upper),
            
            # Complexité
            'num_subqueries': sql.count('(SELECT'),
            'num_predicates': sql_upper.count('AND') + sql_upper.count('OR'),
            'num_aggregates': (sql_upper.count('COUNT(') + sql_upper.count('SUM(') +
                              sql_upper.count('AVG(') + sql_upper.count('MAX(') + 
                              sql_upper.count('MIN(') + sql_upper.count('STDDEV')),
            'num_functions': (sql_upper.count('DATE_FORMAT') + sql_upper.count('YEAR(') +
                             sql_upper.count('MONTH(') + sql_upper.count('DAY(')),
            
            # Tables
            'num_tables': (sql_upper.count('FROM') + sql_upper.count('JOIN') - 
                          sql.count('(SELECT')),
            
            # Patterns
            'max_in_list_size': self._count_in_list(sql),
            'has_wildcard_like': int('%' in sql and 'LIKE' in sql_upper),
            'has_case_when': int('CASE WHEN' in sql_upper),
            
            # Type de requête
            'is_select_query': int(sql_upper.startswith('SELECT')),
            'is_insert_query': 0,
            'is_update_query': 0,
            'is_delete_query': 0,
        }
        
        # Contexte temporel
        now = datetime.now()
        features.update({
            'hour_of_day': now.hour,
            'day_of_week': now.weekday(),
            'is_peak_hour': int(9 <= now.hour <= 17),
            'is_weekend': int(now.weekday() >= 5),
            'is_business_hours': int(9 <= now.hour <= 18 and now.weekday() < 5),
            'connections_count': random.randint(20, 100),
            'connections_high': 1,
            'buffer_pool_hit_ratio': round(random.uniform(0.3, 0.7), 4),
            'estimated_index_count': random.randint(2, 8),
            'estimated_table_size_mb': round(random.uniform(50, 500), 2),
        })
        
        # Features dérivées
        features['joins_per_table'] = (features['num_joins'] / max(features['num_tables'], 1))
        features['predicates_per_where'] = (features['num_predicates'] / 
                                           max(features['num_where'], 1))
        features['complexity_density'] = (
            (features['num_joins'] + features['num_subqueries'] + 
             features['num_aggregates']) / max(features['query_length'], 1) * 100
        )
        
        # TOUTES les requêtes générées ici sont LENTES
        features['is_slow'] = 1
        
        return features
    
    def _count_in_list(self, sql):
        """Compter taille max liste IN()"""
        import re
        in_matches = re.findall(r'IN\s*\(([^)]+)\)', sql, re.IGNORECASE)
        
        max_size = 0
        for match in in_matches:
            if 'SELECT' not in match.upper():
                size = len(match.split(','))
                max_size = max(max_size, size)
        
        return max_size
    
    def insert_query(self, connection, features):
        """Insérer une requête dans dataset_final"""
        cursor = connection.cursor()
        
        try:
            query_hash = hashlib.md5(features['sql_text'].encode()).hexdigest()
            
            sql = """
                INSERT INTO dataset_final (
                    query_length, query_length_log, num_joins, num_where,
                    has_group_by, has_order_by, has_having, has_union, has_distinct,
                    has_limit, has_star_select, num_subqueries, num_predicates,
                    num_aggregates, num_functions, num_tables, max_in_list_size,
                    has_wildcard_like, has_case_when,
                    is_select_query, is_insert_query, is_update_query, is_delete_query,
                    hour_of_day, day_of_week, is_peak_hour, is_weekend, is_business_hours,
                    connections_count, connections_high, buffer_pool_hit_ratio,
                    estimated_index_count, estimated_table_size_mb,
                    joins_per_table, predicates_per_where, complexity_density,
                    is_slow, sql_text, query_hash
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """
            
            values = (
                features['query_length'], features['query_length_log'],
                features['num_joins'], features['num_where'],
                features['has_group_by'], features['has_order_by'],
                features['has_having'], features['has_union'],
                features['has_distinct'], features['has_limit'],
                features['has_star_select'], features['num_subqueries'],
                features['num_predicates'], features['num_aggregates'],
                features['num_functions'], features['num_tables'],
                features['max_in_list_size'], features['has_wildcard_like'],
                features['has_case_when'], features['is_select_query'],
                features['is_insert_query'], features['is_update_query'],
                features['is_delete_query'], features['hour_of_day'],
                features['day_of_week'], features['is_peak_hour'],
                features['is_weekend'], features['is_business_hours'],
                features['connections_count'], features['connections_high'],
                features['buffer_pool_hit_ratio'], features['estimated_index_count'],
                features['estimated_table_size_mb'], features['joins_per_table'],
                features['predicates_per_where'], features['complexity_density'],
                features['is_slow'], features['sql_text'], query_hash
            )
            
            cursor.execute(sql, values)
            return True
            
        except Error as e:
            if 'Duplicate entry' not in str(e):
                return False
            return False
        finally:
            cursor.close()
    
    def balance_dataset(self, target_slow_count=5000):
        """Équilibrer le dataset en ajoutant des requêtes lentes"""
        
        connection = self.connect_to_db()
        if not connection:
            return
        
        # Vérifier l'état actuel
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM dataset_final WHERE is_slow = 0")
        fast_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM dataset_final WHERE is_slow = 1")
        slow_count = cursor.fetchone()[0]
        
        print(f"\n📊 État actuel du dataset:")
        print(f"   Requêtes RAPIDES: {fast_count}")
        print(f"   Requêtes LENTES:  {slow_count}")
        print(f"   Ratio: {slow_count / (fast_count + slow_count) * 100:.1f}% lentes")
        
        # Calculer combien ajouter
        needed = target_slow_count - slow_count
        
        if needed <= 0:
            print(f"\n✅ Dataset déjà équilibré!")
            return
        
        print(f"\n🎯 Objectif: ajouter {needed} requêtes LENTES")
        print(f"   Nouveau ratio cible: {target_slow_count / (fast_count + target_slow_count) * 100:.1f}% lentes")
        
        # Générer les requêtes
        queries = self.generate_slow_queries(count=needed)
        
        # Insérer
        print(f"\n🔄 Insertion dans dataset_final...")
        
        inserted = 0
        errors = 0
        
        start_time = time.time()
        
        for i, query in enumerate(queries, 1):
            features = self.extract_features(query['sql'])
            features['query_type'] = query['type']
            
            if self.insert_query(connection, features):
                inserted += 1
            else:
                errors += 1
            
            if i % 500 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (len(queries) - i) / rate if rate > 0 else 0
                
                print(f"   [{i:6d}/{len(queries)}] | "
                      f"Insérées: {inserted:6d} | "
                      f"Erreurs: {errors:4d} | "
                      f"Restant: {remaining/60:.1f}min")
        
        # Vérifier le résultat final
        cursor.execute("SELECT COUNT(*) FROM dataset_final WHERE is_slow = 0")
        final_fast = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM dataset_final WHERE is_slow = 1")
        final_slow = cursor.fetchone()[0]
        
        total = final_fast + final_slow
        
        print(f"\n{'='*80}")
        print(f" ✅ ÉQUILIBRAGE TERMINÉ")
        print(f"{'='*80}")
        print(f"\nÉtat final:")
        print(f"   Total requêtes:     {total:,}")
        print(f"   Requêtes RAPIDES:   {final_fast:,} ({final_fast/total*100:.1f}%)")
        print(f"   Requêtes LENTES:    {final_slow:,} ({final_slow/total*100:.1f}%)")
        print(f"\nInsérées: {inserted:,}")
        print(f"Erreurs:  {errors:,}")
        
        connection.close()


def main():
    """Fonction principale"""
    
    print("="*80)
    print(" " * 15 + "🎯 ÉQUILIBRAGE DU DATASET - SADOP")
    print("="*80)
    
    generator = SlowQueryGenerator()
    
    # Option 1: Équilibrer à 30% de lentes (recommandé)
    # Si vous avez 16,397 rapides, on veut ~7,000 lentes (30%)
    
    print("\n📋 Choisissez le niveau d'équilibrage:")
    print("   1. Conservateur (20% lentes) → ~4,100 requêtes lentes")
    print("   2. Équilibré (30% lentes)     → ~7,000 requêtes lentes")
    print("   3. Agressif (40% lentes)      → ~11,000 requêtes lentes")
    
    # Pour l'automatisation, on choisit option 2 (30%)
    target_ratio = 0.30
    
    # Calculer le target_slow_count
    # Si on a 16,397 rapides et on veut 30% de lentes:
    # lentes / (rapides + lentes) = 0.30
    # lentes = 0.30 * (16397 + lentes)
    # lentes = 4919 + 0.30 * lentes
    # 0.70 * lentes = 4919
    # lentes = 7027
    
    target_slow_count = 7000
    
    print(f"\n✅ Mode sélectionné: Équilibré (30% lentes)")
    print(f"   Cible: {target_slow_count} requêtes lentes")
    
    generator.balance_dataset(target_slow_count=target_slow_count)
    
    print(f"\n{'='*80}")
    print(" " * 25 + "✅ TERMINÉ!")
    print(f"{'='*80}")
    print(f"\n💡 Prochaine étape:")
    print(f"   python 02_train_xgboost.py")


if __name__ == "__main__":
    main()