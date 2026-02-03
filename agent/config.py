MYSQL_CONFIG = {
    'host': 'localhost',      # localhost quand tu es sur Windows
    'port': 3308,             # Port exposé par Docker
    'user': 'apbd_user',
    'password': 'apbd_pass',
    'database': 'pos',
    'autocommit': True
}

# RL parameters
MAX_INDEXES = 5
EPISODE_LENGTH = 25
INDEX_CREATION_COST = 0.02
INDEX_DROP_PENALTY = 0.01

print(f"🔧 Mode LOCAL - MySQL: {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}")