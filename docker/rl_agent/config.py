import os

MYSQL_CONFIG = {
    'host': 'mysql',          # Nom du service Docker
    'port': 3306,             # Port INTERNE Docker
    'user': os.getenv('MYSQL_USER', 'apbd_user'),
    'password': os.getenv('MYSQL_PASSWORD', 'apbd_pass'),
    'database': os.getenv('MYSQL_DATABASE', 'pos'),
    'autocommit': True
}

# RL parameters
MAX_INDEXES = 5
EPISODE_LENGTH = 25
INDEX_CREATION_COST = 0.02
INDEX_DROP_PENALTY = 0.01

print(f"🐳 Mode DOCKER - MySQL: {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}")