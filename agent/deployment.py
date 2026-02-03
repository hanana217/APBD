import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import DQN
from env_enhanced import MySQLIndexEnvFinal
import time

class IndexOptimizationAgent:
    def __init__(self, model_path="index_optimizer_enhanced"):
        self.model = DQN.load(model_path)
        self.env = MySQLIndexEnvFinal()
        self.current_indexes = 0
        
    def optimize(self, steps=10):
        """Run optimization for specified number of steps"""
        print("Starting index optimization...")
        obs, _ = self.env.reset()
        
        for step in range(steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, _, _, _ = self.env.step(action)
            
            current_indexes = int(obs[0] * self.env.max_indexes)
            print(f"Optimization step {step+1}: "
                  f"Action={['NOOP', 'CREATE', 'DROP'][action]}, "
                  f"Reward={reward:.4f}, "
                  f"Indexes={current_indexes}/{self.env.max_indexes}, "
                  f"Query time={obs[1]:.4f}s")
            
            self.current_indexes = current_indexes
        
        print(f"\nOptimization complete! Final state:")
        print(f"  - Indexes: {self.current_indexes}/{self.env.max_indexes}")
        print(f"  - Query time: {obs[1]:.4f}s")
        
        return self.current_indexes, obs[1]
    
    def get_recommendations(self):
        """Get index recommendations based on current state"""
        print("\nIndex Optimization Recommendations:")
        print("=" * 50)
        print(f"Current indexes: {self.current_indexes}/{self.env.max_indexes}")
        
        if self.current_indexes < 2:
            print("✓ RECOMMENDATION: Create 1-2 indexes on frequently queried columns")
            print("  Suggested indexes:")
            print("  1. CREATE INDEX idx_orders_client_date ON orders(client_id, orderdate)")
            print("  2. CREATE INDEX idx_clients_wilaya ON clients(wilaya)")
        elif self.current_indexes == 2:
            print("✓ OPTIMAL: Maintaining 2 indexes is optimal for your workload")
            print("  No changes needed at this time")
        else:
            print("⚠  CONSIDER: You may have too many indexes")
            print("  Consider dropping less-used indexes")
        
        return self.current_indexes

# Usage
agent = IndexOptimizationAgent()
indexes, query_time = agent.optimize(steps=5)
agent.get_recommendations()