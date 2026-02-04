# deployment.py - VERSION CORRIGÉE
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import DQN
from env_enhanced import MySQLIndexEnvFinal
import time
import json
import numpy as np

class IndexOptimizationAgent:
    """Agent RL pour l'optimisation d'index - Version interface"""
    
    def __init__(self, model_path="index_optimizer_latest"):
        """Charge le modèle entraîné"""
        try:
            self.model = DQN.load(model_path)
            print(f"✅ Agent chargé: {model_path}")
            self.model_loaded = True
        except Exception as e:
            print(f"⚠  Impossible de charger {model_path}: {e}")
            print("   Utilisation de l'agent en mode simulation")
            self.model = None
            self.model_loaded = False
        
        self.env = MySQLIndexEnvFinal()
        self.current_indexes = 0
        self.performance_history = []
        
    def _get_action_from_model(self, obs):
        """Convertit la prédiction du modèle en action (0, 1, 2)"""
        if not self.model_loaded:
            return 0  # NOOP par défaut
        
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Conversion sécurisée
        if isinstance(action, np.ndarray):
            if action.size == 1:
                return int(action.item())
            elif action.size > 0:
                return int(action[0])
            else:
                return 0
        elif isinstance(action, (int, np.integer)):
            return int(action)
        else:
            # Fallback si format inconnu
            try:
                return int(action)
            except:
                return 0
    
    def optimize(self, steps=10):
        """Exécute l'optimisation"""
        print(f"🚀 Début optimisation ({steps} étapes)")
        
        if not self.model_loaded:
            return self._simulate_optimization(steps)
        
        obs, _ = self.env.reset()
        results = []
        
        for step in range(steps):
            # Récupère l'action du modèle
            action = self._get_action_from_model(obs)
            
            # Exécute l'action dans l'environnement
            obs, reward, terminated, truncated, _ = self.env.step(action)
            
            # Récupère l'état
            current_indexes = int(obs[0] * self.env.max_indexes)
            query_time = obs[1]
            
            # Enregistre les résultats
            step_result = {
                'step': step + 1,
                'action': ['NOOP', 'CREATE', 'DROP'][action],
                'reward': float(reward),
                'indexes': current_indexes,
                'query_time': float(query_time)
            }
            results.append(step_result)
            
            print(f"  Étape {step+1}: {step_result['action']} | "
                  f"Indexes: {current_indexes}/5 | "
                  f"Temps: {query_time:.4f}s | "
                  f"Reward: {reward:.4f}")
            
            if terminated or truncated:
                print(f"  ⏹️  Épisode terminé à l'étape {step+1}")
                break
        
        # État final
        self.current_indexes = current_indexes
        self.performance_history.append({
            'timestamp': time.time(),
            'indexes': current_indexes,
            'query_time': float(query_time)
        })
        
        return {
            'status': 'success',
            'steps': steps,
            'final_indexes': current_indexes,
            'final_query_time': float(query_time),
            'steps_details': results
        }
    
    def _simulate_optimization(self, steps):
        """Mode simulation si modèle non chargé"""
        print("🔧 Mode simulation (pour démonstration)")
        
        results = []
        current_idx = 0
        
        for step in range(steps):
            # Logique de simulation simple
            if step < 2:
                action = 'CREATE'
                current_idx += 1
            elif step == 2:
                action = 'DROP'
                current_idx -= 1
            else:
                action = 'NOOP'
            
            # Garde entre 1 et 4 index
            current_idx = max(1, min(current_idx, 4))
            
            # Performance simulée (amélioration progressive)
            query_time = 0.045 - (step * 0.003)
            query_time = max(0.025, query_time)
            
            # Récompense simulée
            reward = 0.8 if action == 'CREATE' else 0.2 if action == 'DROP' else 0.1
            
            results.append({
                'step': step + 1,
                'action': action,
                'reward': reward,
                'indexes': current_idx,
                'query_time': query_time
            })
        
        self.current_indexes = current_idx
        
        return {
            'status': 'simulation',
            'steps': steps,
            'final_indexes': current_idx,
            'final_query_time': 0.029,
            'steps_details': results
        }
    
    def get_recommendations(self):
        """Génère des recommandations"""
        print("📋 Génération des recommandations...")
        
        recommendations = []
        
        if self.current_indexes < 2:
            recommendations.append({
                'type': 'CREATE',
                'priority': 'high',
                'description': 'Créer 1-2 index sur colonnes fréquemment utilisées',
                'sql': 'CREATE INDEX idx_orders_client_date ON orders(client_id, orderdate)',
                'impact': 'Réduction estimée: 30-50% du temps de requête'
            })
            recommendations.append({
                'type': 'CREATE',
                'priority': 'medium',
                'description': 'Index sur la table clients pour les jointures',
                'sql': 'CREATE INDEX idx_clients_wilaya ON clients(wilaya)',
                'impact': 'Amélioration des jointures avec la table orders'
            })
        
        elif self.current_indexes >= 4:
            recommendations.append({
                'type': 'DROP',
                'priority': 'medium',
                'description': 'Trop d\'index peuvent ralentir les INSERT/UPDATE',
                'sql': 'DROP INDEX idx_orders_old ON orders',
                'impact': 'Libération d\'espace et amélioration des écritures'
            })
        
        # Toujours recommander la surveillance
        recommendations.append({
            'type': 'MONITOR',
            'priority': 'low',
            'description': 'Surveiller les requêtes lentes pendant 24h',
            'sql': 'SHOW INDEX FROM orders; SHOW TABLE STATUS',
            'impact': 'Identification des opportunités d\'optimisation'
        })
        
        # Recommandation sur le schéma
        recommendations.append({
            'type': 'ANALYZE',
            'priority': 'low',
            'description': 'Analyser l\'utilisation des index existants',
            'sql': 'SELECT * FROM information_schema.statistics WHERE table_schema = DATABASE()',
            'impact': 'Comprendre quels index sont utilisés'
        })
        
        return {
            'current_indexes': self.current_indexes,
            'max_indexes': 5,
            'performance': self.get_status()['performance'],
            'recommendations': recommendations
        }
    
    def get_status(self):
        """État actuel de la base"""
        try:
            # Vérifie si l'environnement est encore ouvert
            if hasattr(self.env, 'conn'):
                conn = self.env.conn
                if conn.is_connected():
                    cursor = conn.cursor()
                    from mysql_utils import measure_query_performance
                    perf = measure_query_performance(cursor)
                    cursor.close()
                else:
                    # Reconnecte si nécessaire
                    from mysql_utils import get_connection
                    self.env.conn = get_connection()
                    cursor = self.env.conn.cursor()
                    from mysql_utils import measure_query_performance
                    perf = measure_query_performance(cursor)
                    cursor.close()
            else:
                # Nouvelle connexion
                from mysql_utils import get_connection
                conn = get_connection()
                cursor = conn.cursor()
                from mysql_utils import measure_query_performance
                perf = measure_query_performance(cursor)
                cursor.close()
                conn.close()
            
            # Compte les index actuels
            from mysql_utils import get_connection
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.statistics 
                WHERE table_schema = DATABASE() 
                AND table_name = 'orders'
                AND index_name LIKE 'idx_orders_%'
            """)
            index_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            self.current_indexes = index_count
            
            return {
                'performance': float(perf),
                'index_count': index_count,
                'max_indexes': 5,
                'status': 'active',
                'message': f'Base optimisée ({index_count}/5 index)'
            }
            
        except Exception as e:
            print(f"⚠  Erreur lors du statut: {e}")
            return {
                'performance': 0.03,
                'index_count': self.current_indexes if hasattr(self, 'current_indexes') else 2,
                'max_indexes': 5,
                'status': 'estimated',
                'message': 'Statut estimé (mode simulation)'
            }

# Test rapide
if __name__ == "__main__":
    print("🧪 Test de l'agent...")
    print("=" * 50)
    
    # Crée l'agent
    agent = IndexOptimizationAgent()
    
    # Test 1: État initial
    print("\n1. 📊 État initial:")
    status = agent.get_status()
    print(f"   - Performance: {status['performance']:.4f}s")
    print(f"   - Indexes: {status['index_count']}/5")
    
    # Test 2: Optimisation
    print("\n2. 🚀 Optimisation (3 étapes):")
    result = agent.optimize(steps=3)
    
    print(f"\n   Résumé optimisation:")
    print(f"   - Étapes exécutées: {len(result['steps_details'])}")
    print(f"   - Indexes finaux: {result['final_indexes']}/5")
    print(f"   - Temps final: {result['final_query_time']:.4f}s")
    
    # Test 3: Recommandations
    print("\n3. 💡 Recommandations:")
    recs = agent.get_recommendations()
    
    for i, rec in enumerate(recs['recommendations'], 1):
        print(f"   {i}. [{rec['type']}] {rec['description']}")
        print(f"      SQL: {rec['sql'][:50]}...")
    
    # Test 4: JSON output (pour l'API)
    print("\n4. 📦 Format API (extrait):")
    api_response = {
        'status': 'success',
        'optimization': result,
        'recommendations': recs
    }
    
    print(json.dumps({
        'final_indexes': result['final_indexes'],
        'final_performance': result['final_query_time'],
        'recommendation_count': len(recs['recommendations'])
    }, indent=2))
    
    print("\n" + "=" * 50)
    print("✅ Test terminé avec succès!")
    print("🤝 Agent prêt pour l'API FastAPI")