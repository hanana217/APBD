import os

MYSQL_CONFIG = {
    'host': os.environ.get('MYSQL_HOST', 'localhost'),
    'port': int(os.environ.get('MYSQL_PORT', '3306')),
    'user': os.environ.get('MYSQL_USER', 'apbd_user'),
    'password': os.environ.get('MYSQL_PASSWORD', 'apbd_pass'),
    'database': os.environ.get('MYSQL_DATABASE', 'pos'),
    'autocommit': True
}

# RL parameters
MAX_INDEXES = 5
EPISODE_LENGTH = 25
INDEX_CREATION_COST = 0.02
INDEX_DROP_PENALTY = 0.01

print(f"🔧 MySQL: {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}")