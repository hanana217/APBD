# config_enhanced.py
"""
Enhanced configuration for MySQL RL Agent with binary matrix representation
KEYWORD: ENHANCED_CONFIG
"""

MYSQL_CONFIG = {
    'host': 'localhost',
    'port': 3308,
    'user': 'apbd_user',
    'password': 'apbd_pass',
    'database': 'pos',
    'autocommit': True
}

# Enhanced RL parameters
MAX_INDEXES = 5
EPISODE_LENGTH = 25
INDEX_CREATION_COST = 0.02
INDEX_DROP_PENALTY = 0.01
INSERT_IMPACT_WEIGHT = 150  # Weight for INSERT slowdown penalty

# Database schema definition for binary matrix
SCHEMA_DEFINITION = {
    'orders': {
        'columns': ['id', 'client_id', 'orderdate', 'price', 'cart_id', 'status'],
        'primary_key': ['id'],
        'foreign_keys': [
            {'column': 'client_id', 'references': 'clients.id'},
            {'column': 'cart_id', 'references': 'cart.id'}
        ],
        'size_mb': 250,
        'row_count': 1000000
    },
    'clients': {
        'columns': ['id', 'name', 'wilaya', 'address', 'phone', 'email'],
        'primary_key': ['id'],
        'size_mb': 50,
        'row_count': 50000
    },
    'products': {
        'columns': ['id', 'name', 'category', 'price', 'stock'],
        'primary_key': ['id'],
        'size_mb': 100,
        'row_count': 20000
    }
}

# Column importance weights (learned or predefined)
COLUMN_IMPORTANCE = {
    'orders.client_id': 0.9,
    'orders.orderdate': 0.8,
    'orders.price': 0.6,
    'clients.wilaya': 0.7,
    'clients.address': 0.4,
    'products.category': 0.5,
    'products.price': 0.6
}

# Workload patterns
WORKLOAD_PATTERNS = {
    'read_heavy': 0.7,      # 70% reads
    'write_heavy': 0.2,     # 20% writes
    'mixed': 0.1            # 10% mixed
}

print("🔧 ENHANCED MODE - Binary matrix representation activated")
print(f"📊 Schema: {len(SCHEMA_DEFINITION)} tables, {sum(len(t['columns']) for t in SCHEMA_DEFINITION.values())} columns")