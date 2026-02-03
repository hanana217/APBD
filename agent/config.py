MYSQL_CONFIG = {
    'host': '127.0.0.1',
    'port': 3308,
    'user': 'apbd_user',
    'password': 'apbd_pass',
    'database': 'pos'
}

# RL parameters
MAX_INDEXES = 5
EPISODE_LENGTH = 25  # Shorter episodes
INDEX_CREATION_COST = 0.02  # Higher cost to prevent over-indexing
INDEX_DROP_PENALTY = 0.01
#INDEX_OVERFLOW_PENALTY = 0.5  # Heavy penalty for exceeding max indexes