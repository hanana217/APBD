# agent_enhanced.py
"""
Enhanced RL Agent with binary matrix understanding
KEYWORD: ENHANCED_AGENT
"""

import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from env_enhanced import MySQLIndexEnvEnhanced
import numpy as np
import time
import json
import os

class EnhancedTrainingCallback(BaseCallback):
    """Enhanced callback for monitoring training"""
    
    def __init__(self, check_freq=100, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.best_mean_reward = -np.inf
        self.performance_history = []
    
    def _on_step(self):
        # Track rewards
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Episode ended
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Log every 3 episodes
            if len(self.episode_rewards) % 3 == 0:
                mean_reward = np.mean(self.episode_rewards[-5:]) if len(self.episode_rewards) >= 5 else np.mean(self.episode_rewards)
                mean_length = np.mean(self.episode_lengths[-5:]) if len(self.episode_lengths) >= 5 else np.mean(self.episode_lengths)
                
                print(f"\n📊 Episode {len(self.episode_rewards)}")
                print(f"  Reward: {self.current_episode_reward:.2f}")
                print(f"  Mean reward (last 5): {mean_reward:.2f}")
                print(f"  Length: {self.current_episode_length}")
                
                # Save if improved
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save("index_optimizer_enhanced")
                    print(f"  💾 New best model saved!")
            
            # Reset
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Periodic checkpoint
        if self.n_calls % 500 == 0:
            self.model.save(f"index_optimizer_enhanced_checkpoint_{self.n_calls}")
        
        return True

class EnhancedIndexOptimizationAgent:
    """
    Enhanced RL Agent for MySQL Index Optimization
    - Understands binary matrix representation
    - Considers INSERT impact
    - Learns column-specific strategies
    """
    
    def __init__(self, model_path="index_optimizer_latest"):
        print("="*70)
        print("🤖 ENHANCED INDEX OPTIMIZATION AGENT")
        print("="*70)
        
        # Try to load enhanced model
        self.model = None
        self.model_loaded = False
        
        model_paths = [
            model_path,
            "index_optimizer_enhanced_best",
            "index_optimizer_enhanced",
            "index_optimizer_latest",
            "index_optimizer_final"
        ]
        
        for path in model_paths:
            try:
                self.model = DQN.load(path)
                self.model_loaded = True
                print(f"✅ Enhanced model loaded: {path}")
                break
            except:
                continue
        
        if not self.model_loaded:
            print("⚠️ No enhanced model found - will use simulation mode")
        
        # Initialize environment
        try:
            self.env = MySQLIndexEnvEnhanced()
            print("✅ Enhanced environment connected")
        except Exception as e:
            print(f"❌ Environment error: {e}")
            self.env = None
        
        # State tracking
        self.current_indexes = 0
        self.performance_history = []
        self.binary_matrix_history = []
        
        print("="*70)
    
    def _interpret_binary_matrix(self, binary_matrix):
        """
        Interpret binary matrix state for human-readable output
        KEYWORD: INTERPRET_BINARY_MATRIX
        """
        if not hasattr(self, 'env') or not self.env:
            return "Environment not available"
        
        interpretation = []
        col_idx = 0
        
        try:
            from config import SCHEMA_DEFINITION
            
            for table, schema in SCHEMA_DEFINITION.items():
                for column in schema['columns']:
                    if col_idx < len(binary_matrix):
                        is_indexed = binary_matrix[col_idx] > 0.5
                        status = "✅ INDEXED" if is_indexed else "❌ NOT INDEXED"
                        
                        # Get column importance
                        col_key = f"{table}.{column}"
                        importance = 0.3
                        if hasattr(self.env, 'column_importance') and col_key in self.env.column_importance:
                            importance = self.env.column_importance[col_key]
                        
                        interpretation.append({
                            'table': table,
                            'column': column,
                            'indexed': bool(is_indexed),
                            'importance': importance,
                            'recommendation': "Keep" if is_indexed and importance > 0.5 else "Consider" if importance > 0.6 else "Monitor"
                        })
                    
                    col_idx += 1
            
            return interpretation
            
        except Exception as e:
            return f"Interpretation error: {e}"
    
    def optimize(self, steps=10, strategy='balanced'):
        """
        Enhanced optimization with binary matrix awareness
        """
        print(f"\n{'='*70}")
        print(f"🚀 ENHANCED OPTIMIZATION ({steps} steps, strategy: {strategy})")
        print(f"{'='*70}\n")
        
        if not self.model_loaded or not self.env:
            return self._simulate_enhanced_optimization(steps, strategy)
        
        try:
            obs, _ = self.env.reset()
            results = []
            binary_matrices = []
            
            for step in range(steps):
                # Predict action
                action, _ = self.model.predict(obs, deterministic=True)
                action = int(action)
                
                # Execute
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Extract and interpret binary matrix
                binary_matrix = obs[:self.env.total_columns]
                matrix_interpretation = self._interpret_binary_matrix(binary_matrix)
                
                # Count indexed columns
                indexed_count = int(np.sum(binary_matrix > 0.5))
                
                # Store results
                step_result = {
                    'step': step + 1,
                    'action': ['NOOP', 'CREATE', 'DROP'][action],
                    'reward': float(reward),
                    'indexed_columns': indexed_count,
                    'select_perf': float(info.get('select_perf', 0)),
                    'insert_perf': float(info.get('insert_perf', 0)),
                    'binary_matrix': binary_matrix.tolist(),
                    'matrix_interpretation': matrix_interpretation
                }
                
                if 'action_details' in info:
                    step_result['action_details'] = info['action_details']
                
                results.append(step_result)
                binary_matrices.append(binary_matrix)
                
                # Stop if episode ended
                if terminated or truncated:
                    print(f"  ⏹️ Optimization ended at step {step+1}")
                    break
            
            # Final analysis
            final_indexed = results[-1]['indexed_columns'] if results else 0
            final_select = results[-1]['select_perf'] if results else 0
            final_insert = results[-1]['insert_perf'] if results else 0
            
            # Calculate improvement
            initial_select = results[0]['select_perf'] if results else 0
            improvement_pct = ((initial_select - final_select) / initial_select * 100) if initial_select > 0 else 0
            
            print(f"\n{'='*70}")
            print("📊 ENHANCED OPTIMIZATION RESULTS")
            print(f"{'='*70}")
            print(f"Final indexed columns: {final_indexed}")
            print(f"SELECT performance: {final_select:.4f}s")
            print(f"INSERT performance: {final_insert:.4f}s")
            print(f"Improvement: {improvement_pct:+.1f}%")
            
            # Binary matrix analysis
            if binary_matrices:
                print(f"\n📈 Binary Matrix Evolution:")
                initial_indexed = np.sum(binary_matrices[0] > 0.5)
                final_indexed = np.sum(binary_matrices[-1] > 0.5)
                print(f"  Initial indexed columns: {initial_indexed}")
                print(f"  Final indexed columns: {final_indexed}")
                print(f"  Change: {final_indexed - initial_indexed:+d}")
            
            return {
                'status': 'success',
                'mode': 'enhanced_rl',
                'steps': len(results),
                'final_indexed_columns': final_indexed,
                'final_select_perf': final_select,
                'final_insert_perf': final_insert,
                'improvement_pct': improvement_pct,
                'steps_details': results,
                'binary_matrix_analysis': {
                    'initial': binary_matrices[0].tolist() if binary_matrices else [],
                    'final': binary_matrices[-1].tolist() if binary_matrices else []
                }
            }
            
        except Exception as e:
            print(f"❌ Enhanced optimization error: {e}")
            return self._simulate_enhanced_optimization(steps, strategy)
    
    def _simulate_enhanced_optimization(self, steps=10, strategy='balanced'):
        """Simulated enhanced optimization"""
        print("🔧 SIMULATION MODE (Enhanced)")
        print("-"*60)
        
        results = []
        binary_matrix = np.zeros(20, dtype=np.float32)  # Simulated matrix
        
        for step in range(steps):
            # Simulated logic
            if step < 3:
                action = 'CREATE'
                # Simulate indexing important columns
                if step == 0:
                    binary_matrix[2] = 1.0  # orders.client_id
                elif step == 1:
                    binary_matrix[3] = 1.0  # orders.orderdate
                elif step == 2:
                    binary_matrix[8] = 1.0  # clients.wilaya
            elif step == 4:
                action = 'DROP'
                binary_matrix[8] = 0.0  # Drop less important index
            else:
                action = 'NOOP'
            
            indexed_count = int(np.sum(binary_matrix))
            
            # Simulated performance
            select_perf = 0.045 - (step * 0.002)
            insert_perf = 0.012 + (indexed_count * 0.0015)
            
            # Simulated reward
            reward = {
                'CREATE': 0.7,
                'DROP': 0.3,
                'NOOP': 0.1
            }[action]
            
            results.append({
                'step': step + 1,
                'action': action,
                'reward': reward,
                'indexed_columns': indexed_count,
                'select_perf': select_perf,
                'insert_perf': insert_perf,
                'binary_matrix': binary_matrix.tolist()
            })
            
            print(f"  Step {step+1}: {action:6s} | Indexed: {indexed_count:2d} | "
                  f"SELECT: {select_perf:.4f}s | INSERT: {insert_perf:.4f}s")
        
        return {
            'status': 'simulation',
            'mode': 'enhanced_simulation',
            'steps': steps,
            'final_indexed_columns': results[-1]['indexed_columns'],
            'final_select_perf': results[-1]['select_perf'],
            'final_insert_perf': results[-1]['insert_perf'],
            'improvement_pct': 25.0,
            'steps_details': results
        }
    
    def get_enhanced_status(self):
        """Get enhanced status with binary matrix analysis"""
        try:
            if not self.env:
                return {
                    'status': 'simulation',
                    'message': 'Enhanced environment not available',
                    'binary_matrix': []
                }
            
            # Get current state
            obs, _ = self.env.reset()
            binary_matrix = obs[:self.env.total_columns]
            
            # Interpret matrix
            interpretation = self._interpret_binary_matrix(binary_matrix)
            
            # Count statistics
            indexed_count = int(np.sum(binary_matrix > 0.5))
            total_columns = len(binary_matrix)
            
            # Performance
            select_perf = measure_query_performance(self.env.cursor)
            insert_perf = measure_insert_performance(self.env.cursor)
            
            return {
                'status': 'active',
                'indexed_columns': indexed_count,
                'total_columns': total_columns,
                'indexing_ratio': indexed_count / total_columns if total_columns > 0 else 0,
                'select_performance': float(select_perf),
                'insert_performance': float(insert_perf),
                'binary_matrix': binary_matrix.tolist(),
                'matrix_interpretation': interpretation,
                'message': f'Enhanced agent active - {indexed_count}/{total_columns} columns indexed'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Enhanced status error: {str(e)}',
                'binary_matrix': []
            }
    
    def get_enhanced_recommendations(self):
        """Get enhanced recommendations based on binary matrix analysis"""
        print("\n💡 ENHANCED RECOMMENDATIONS")
        print("-"*60)
        
        try:
            status = self.get_enhanced_status()
            
            if status['status'] != 'active':
                return self._get_simulation_recommendations()
            
            binary_matrix = np.array(status['binary_matrix'])
            interpretation = status['matrix_interpretation']
            
            recommendations = []
            
            # Analyze current state
            indexed_count = status['indexed_columns']
            indexing_ratio = status['indexing_ratio']
            
            # 1. Overall strategy recommendation
            if indexing_ratio < 0.2:
                recommendations.append({
                    'type': 'STRATEGY',
                    'priority': 'high',
                    'title': 'Increase Index Coverage',
                    'description': f'Only {indexed_count} columns are indexed ({indexing_ratio:.0%}). Consider adding indexes to frequently queried columns.',
                    'action': 'CREATE on high-importance columns'
                })
            elif indexing_ratio > 0.4:
                recommendations.append({
                    'type': 'STRATEGY',
                    'priority': 'medium',
                    'title': 'Optimize Existing Indexes',
                    'description': f'{indexed_count} columns indexed ({indexing_ratio:.0%}). Focus on optimizing rather than adding more.',
                    'action': 'Consider DROP on low-usage indexes'
                })
            
            # 2. Column-specific recommendations
            if isinstance(interpretation, list):
                high_importance_unindexed = [
                    item for item in interpretation 
                    if not item['indexed'] and item['importance'] > 0.6
                ]
                
                low_importance_indexed = [
                    item for item in interpretation 
                    if item['indexed'] and item['importance'] < 0.4
                ]
                
                if high_importance_unindexed:
                    for item in high_importance_unindexed[:2]:
                        recommendations.append({
                            'type': 'CREATE',
                            'priority': 'high',
                            'title': f'Index {item["table"]}.{item["column"]}',
                            'description': f'High importance column ({item["importance"]:.2f}) is not indexed.',
                            'sql': f'CREATE INDEX idx_{item["table"]}_{item["column"]} ON {item["table"]}({item["column"]})',
                            'impact': 'Expected SELECT improvement: 20-40%'
                        })
                
                if low_importance_indexed:
                    for item in low_importance_indexed[:2]:
                        recommendations.append({
                            'type': 'DROP',
                            'priority': 'medium',
                            'title': f'Review index on {item["table"]}.{item["column"]}',
                            'description': f'Low importance column ({item["importance"]:.2f}) is indexed.',
                            'sql': f'DROP INDEX idx_{item["table"]}_{item["column"]} ON {item["table"]}',
                            'impact': 'Expected INSERT improvement: 5-15%'
                        })
            
            # 3. Performance-based recommendations
            if status['insert_performance'] > 0.02:  # INSERT > 20ms
                recommendations.append({
                    'type': 'MONITOR',
                    'priority': 'medium',
                    'title': 'INSERT Performance Alert',
                    'description': f'INSERT performance is {status["insert_performance"]*1000:.1f}ms. Consider if too many indexes are slowing writes.',
                    'action': 'Monitor INSERT times and consider index consolidation'
                })
            
            # 4. General best practices
            recommendations.append({
                'type': 'BEST_PRACTICE',
                'priority': 'low',
                'title': 'Regular Index Maintenance',
                'description': 'Regularly analyze index usage and remove unused indexes.',
                'sql': "SELECT * FROM sys.schema_unused_indexes WHERE object_schema = DATABASE()"
            })
            
            print(f"  Generated {len(recommendations)} enhanced recommendations")
            
            return {
                'current_state': {
                    'indexed_columns': indexed_count,
                    'total_columns': status['total_columns'],
                    'select_performance': status['select_performance'],
                    'insert_performance': status['insert_performance']
                },
                'recommendations': recommendations,
                'binary_matrix_summary': {
                    'total_columns': len(binary_matrix),
                    'indexed_count': indexed_count,
                    'indexing_ratio': indexing_ratio
                }
            }
            
        except Exception as e:
            print(f"  Recommendation error: {e}")
            return self._get_simulation_recommendations()
    
    def _get_simulation_recommendations(self):
        """Fallback simulation recommendations"""
        return {
            'current_state': {
                'indexed_columns': 2,
                'total_columns': 20,
                'select_performance': 0.035,
                'insert_performance': 0.015
            },
            'recommendations': [
                {
                    'type': 'CREATE',
                    'priority': 'high',
                    'title': 'Index orders.client_id',
                    'description': 'Frequently used in WHERE clauses and JOINs',
                    'sql': 'CREATE INDEX idx_orders_client_id ON orders(client_id)',
                    'impact': 'Expected improvement: 30-50%'
                },
                {
                    'type': 'CREATE',
                    'priority': 'medium',
                    'title': 'Index clients.wilaya',
                    'description': 'Common filter in reports',
                    'sql': 'CREATE INDEX idx_clients_wilaya ON clients(wilaya)',
                    'impact': 'Expected improvement: 20-30%'
                }
            ]
        }


# Test the enhanced agent
if __name__ == "__main__":
    print("🧪 Testing Enhanced Agent")
    print("="*60)
    
    agent = EnhancedIndexOptimizationAgent()
    
    # Test status
    print("\n1️⃣ Enhanced Status:")
    status = agent.get_enhanced_status()
    print(f"   Status: {status.get('status', 'unknown')}")
    print(f"   Indexed columns: {status.get('indexed_columns', 'N/A')}")
    
    # Test optimization
    print("\n2️⃣ Test Optimization (5 steps):")
    result = agent.optimize(steps=5)
    print(f"   Mode: {result['mode']}")
    print(f"   Steps: {result['steps']}")
    print(f"   Final indexed: {result.get('final_indexed_columns', 'N/A')}")
    
    # Test recommendations
    print("\n3️⃣ Enhanced Recommendations:")
    recs = agent.get_enhanced_recommendations()
    print(f"   Generated: {len(recs['recommendations'])} recommendations")
    
    for i, rec in enumerate(recs['recommendations'][:2], 1):
        print(f"\n   {i}. [{rec['type']}] {rec['title']}")
        print(f"      Priority: {rec['priority']}")
        print(f"      {rec['description']}")
    
    print("\n✅ Enhanced agent test completed")