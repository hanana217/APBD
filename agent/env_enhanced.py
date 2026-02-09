# env_enhanced_complete.py
"""
Complete enhanced environment with binary matrix representation
KEYWORD: ENHANCED_ENVIRONMENT
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from mysql_utils import *
from config import *
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class MySQLIndexEnvEnhanced(gym.Env):
    """
    Enhanced RL Environment with binary matrix representation
    - Agent sees which specific columns are indexed
    - Agent understands INSERT/UPDATE impact
    - Agent learns column-specific strategies
    """
    
    def __init__(self):
        super().__init__()
        
        print("="*70)
        print("🤖 ENHANCED RL ENVIRONMENT - BINARY MATRIX REPRESENTATION")
        print("="*70)
        
        # Calculate state dimensions
        self.total_columns = sum(len(schema['columns']) 
                                for schema in SCHEMA_DEFINITION.values())
        
        print(f"📊 State dimensions:")
        print(f"  - Binary matrix: {self.total_columns} columns")
        print(f"  - Performance metrics: 3 metrics")
        print(f"  - Workload patterns: 2 patterns")
        print(f"  - Table statistics: {len(SCHEMA_DEFINITION)} tables")
        
        # Enhanced action space
        # Actions: 0=NOOP, 1=CREATE (smart), 2=DROP (smart)
        self.action_space = spaces.Discrete(3)
        
        # Enhanced state space
        state_dim = (
            self.total_columns +      # Binary matrix
            3 +                       # Performance metrics
            2 +                       # Workload patterns
            len(SCHEMA_DEFINITION)    # Table stats
        )
        
        self.observation_space = spaces.Box(
            low=np.zeros(state_dim, dtype=np.float32),
            high=np.ones(state_dim, dtype=np.float32),
            dtype=np.float32
        )
        
        # Database connection
        self.conn = get_connection()
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()
        
        # Tracking variables
        self.query_times = []
        self.insert_times = []
        self.max_indexes = MAX_INDEXES
        self.step_count = 0
        self.best_performance = None
        self.index_history = []
        self.last_actions = []
        
        # Workload tracking
        self.workload_pattern = {'reads': 0, 'writes': 0}
        
        # Column importance (dynamic)
        self.column_importance = COLUMN_IMPORTANCE.copy()
        
        print("✅ Enhanced environment initialized")
        print("="*70)
    
    def _clear_cursor_results(self):
        """Clear any unread results from cursor"""
        try:
            if self.cursor.with_rows:
                self.cursor.fetchall()
        except:
            pass
        try:
            while self.cursor.nextset():
                try:
                    self.cursor.fetchall()
                except:
                    pass
        except:
            pass
    
    def _get_smart_create_target(self):
        """
        Smart CREATE: Choose which column to index based on:
        1. Column importance
        2. Current indexing status
        3. Query patterns
        KEYWORD: SMART_CREATE
        """
        # Get current binary matrix
        binary_matrix = get_binary_index_matrix(self.cursor)
        
        # Find unindexed columns with high importance
        unindexed_scores = []
        
        col_idx = 0
        for table, schema in SCHEMA_DEFINITION.items():
            for col in schema['columns']:
                col_key = f"{table}.{col}"
                
                # Skip primary keys (already indexed)
                if col in schema.get('primary_key', []):
                    col_idx += 1
                    continue
                
                # Check if already indexed
                is_indexed = binary_matrix[col_idx] if col_idx < len(binary_matrix) else 0
                
                if not is_indexed:
                    importance = self.column_importance.get(col_key, 0.3)
                    unindexed_scores.append({
                        'table': table,
                        'column': col,
                        'importance': importance,
                        'matrix_index': col_idx
                    })
                
                col_idx += 1
        
        # Sort by importance and choose top
        if unindexed_scores:
            unindexed_scores.sort(key=lambda x: x['importance'], reverse=True)
            return unindexed_scores[0]  # Most important unindexed column
        
        return None
    
    def _get_smart_drop_target(self):
        """
        Smart DROP: Choose which index to drop based on:
        1. Column importance (lowest first)
        2. Index usage (if available)
        3. Impact on INSERT performance
        KEYWORD: SMART_DROP
        """
        try:
            # Get all existing indexes
            self.cursor.execute("""
                SELECT TABLE_NAME, INDEX_NAME, COLUMN_NAME
                FROM INFORMATION_SCHEMA.STATISTICS
                WHERE TABLE_SCHEMA = DATABASE()
                AND INDEX_NAME != 'PRIMARY'
                AND INDEX_NAME LIKE 'idx_%'
            """)
            
            indexes = self.cursor.fetchall()
            
            if not indexes:
                return None
            
            # Score indexes for potential dropping
            drop_candidates = []
            
            for table, idx_name, column in indexes:
                col_key = f"{table}.{column}"
                importance = self.column_importance.get(col_key, 0.3)
                
                drop_candidates.append({
                    'table': table,
                    'index_name': idx_name,
                    'column': column,
                    'importance': importance,
                    'score': 1 - importance  # Lower importance = higher drop score
                })
            
            # Sort by drop score (highest first)
            drop_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            return drop_candidates[0] if drop_candidates else None
            
        except Exception as e:
            print(f"[Smart Drop Error] {e}")
            return None
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        print("\n" + "="*60)
        print("🔄 RESET ENHANCED ENVIRONMENT")
        print("="*60)
        
        self.step_count = 0
        self.query_times = []
        self.insert_times = []
        self.index_history = []
        self.last_actions = []
        self.workload_pattern = {'reads': 0, 'writes': 0}
        self._clear_cursor_results()
        
        # Clean up test indexes
        try:
            self.cursor.execute("SHOW INDEX FROM orders WHERE Key_name LIKE 'idx_orders_%'")
            indexes = self.cursor.fetchall()
            for idx in indexes:
                try:
                    self.cursor.execute(f"DROP INDEX {idx[2]} ON orders")
                    self._clear_cursor_results()
                except:
                    pass
        except Exception as e:
            print(f"[Reset Cleanup] {e}")
        
        # Measure initial performance
        select_perf = measure_query_performance(self.cursor)
        insert_perf = measure_insert_performance(self.cursor)
        
        self.query_times.append(select_perf)
        self.insert_times.append(insert_perf)
        self.best_performance = select_perf
        
        current_indexes = get_existing_indexes(self.cursor)
        self.index_history.append(current_indexes)
        
        print(f"📊 Initial state:")
        print(f"  - SELECT performance: {select_perf:.4f}s")
        print(f"  - INSERT performance: {insert_perf:.4f}s")
        print(f"  - Index count: {current_indexes}/{self.max_indexes}")
        
        # Get initial state
        state = self._get_state()
        
        print("="*60)
        return state, {}
    
    def _get_state(self):
        """
        Get complete enhanced state representation
        Returns: Binary matrix + performance + workload + table stats
        KEYWORD: ENHANCED_STATE
        """
        state_parts = []
        
        # 1. Binary matrix of indexes
        binary_matrix = get_binary_index_matrix(self.cursor)
        state_parts.append(binary_matrix)
        
        # 2. Performance metrics (normalized)
        select_perf = measure_query_performance(self.cursor)
        insert_perf = measure_insert_performance(self.cursor)
        
        state_parts.append(np.array([
            min(select_perf / 0.1, 1.0),        # SELECT normalized to 100ms
            min(insert_perf / 0.05, 1.0),       # INSERT normalized to 50ms
            (self.query_times[-2] - select_perf) / (self.query_times[-2] + 1e-6) if len(self.query_times) > 1 else 0.0
        ], dtype=np.float32))
        
        # 3. Workload patterns
        total_ops = self.workload_pattern['reads'] + self.workload_pattern['writes']
        if total_ops > 0:
            read_ratio = self.workload_pattern['reads'] / total_ops
            write_ratio = self.workload_pattern['writes'] / total_ops
        else:
            read_ratio = 0.7  # Default read-heavy
            write_ratio = 0.3
        
        state_parts.append(np.array([read_ratio, write_ratio], dtype=np.float32))
        
        # 4. Table statistics (normalized)
        table_stats = get_table_statistics(self.cursor)
        table_stats_vector = []
        
        for table in SCHEMA_DEFINITION.keys():
            if table in table_stats:
                # Normalize rows to 1M, size to 500MB
                rows_norm = min(table_stats[table]['rows'] / 1000000, 1.0)
                size_norm = min(table_stats[table]['size_mb'] / 500, 1.0)
                table_stats_vector.extend([rows_norm, size_norm])
            else:
                table_stats_vector.extend([0.0, 0.0])
        
        state_parts.append(np.array(table_stats_vector, dtype=np.float32))
        
        # Combine all parts
        full_state = np.concatenate(state_parts, dtype=np.float32)
        
        # Ensure correct dimensions
        expected_dim = self.observation_space.shape[0]
        if len(full_state) < expected_dim:
            # Pad with zeros
            full_state = np.pad(full_state, (0, expected_dim - len(full_state)), 
                              mode='constant', constant_values=0.0)
        elif len(full_state) > expected_dim:
            # Truncate
            full_state = full_state[:expected_dim]
        
        return full_state
    
    def _calculate_insert_impact_penalty(self, current_indexes):
        """
        Calculate penalty for INSERT slowdown due to indexes
        Exponential penalty for excessive indexes
        KEYWORD: INSERT_IMPACT_PENALTY
        """
        insert_time = measure_insert_performance(self.cursor)
        
        if len(self.insert_times) > 1:
            baseline_insert = self.insert_times[0]
            insert_degradation = insert_time - baseline_insert
        else:
            insert_degradation = 0
        
        # Base penalty for any degradation
        penalty = max(0, insert_degradation * INSERT_IMPACT_WEIGHT)
        
        # Exponential penalty for excessive indexes
        if current_indexes > self.max_indexes:
            excess = current_indexes - self.max_indexes
            exponential_penalty = (1.5 ** excess) * 0.2
            penalty += exponential_penalty
        
        # Critical penalty for catastrophic behavior (10+ indexes)
        if current_indexes >= 10:
            catastrophic_penalty = 5.0 + (current_indexes - 10) * 1.0
            penalty += catastrophic_penalty
        
        return penalty, insert_time
    
    def step(self, action):
        """
        Execute action in environment
        Enhanced with smart CREATE/DROP and INSERT impact penalties
        """
        self.step_count += 1
        self.workload_pattern['reads'] += 1  # Each step involves reading
        
        current_indexes = get_existing_indexes(self.cursor)
        action_successful = False
        action_blocked = False
        action_details = {}
        
        print(f"\n📈 Step {self.step_count}: Action = {['NOOP', 'CREATE', 'DROP'][action]}")
        print("-"*40)
        
        # Check action feasibility
        if action == 1 and current_indexes >= self.max_indexes:
            action_blocked = True
            print(f"  ⚠️ CREATE blocked: At max indexes ({current_indexes}/{self.max_indexes})")
        
        elif action == 2 and current_indexes <= 0:
            action_blocked = True
            print(f"  ⚠️ DROP blocked: No indexes to drop")
        
        # Execute action if not blocked
        if not action_blocked:
            if action == 1:  # SMART CREATE
                target = self._get_smart_create_target()
                if target:
                    try:
                        index_name = f"idx_{target['table']}_{target['column']}_{int(time.time())}"
                        sql = f"CREATE INDEX {index_name} ON {target['table']}({target['column']})"
                        
                        self.cursor.execute(sql)
                        self._clear_cursor_results()
                        
                        action_successful = True
                        action_details = target
                        print(f"  ✅ Created index on {target['table']}.{target['column']}")
                        print(f"     Importance: {target['importance']:.2f}")
                        
                        self.workload_pattern['writes'] += 1
                        
                    except Exception as e:
                        print(f"  ❌ CREATE failed: {e}")
                else:
                    print(f"  ⚠️ No suitable column found for indexing")
            
            elif action == 2:  # SMART DROP
                target = self._get_smart_drop_target()
                if target:
                    try:
                        sql = f"DROP INDEX {target['index_name']} ON {target['table']}"
                        self.cursor.execute(sql)
                        self._clear_cursor_results()
                        
                        action_successful = True
                        action_details = target
                        print(f"  ✅ Dropped index {target['index_name']}")
                        print(f"     Column: {target['table']}.{target['column']}")
                        print(f"     Importance: {target['importance']:.2f}")
                        
                        self.workload_pattern['writes'] += 1
                        
                    except Exception as e:
                        print(f"  ❌ DROP failed: {e}")
                else:
                    print(f"  ⚠️ No suitable index found to drop")
            
            else:  # NOOP
                print(f"  ⏸️ No operation")
        
        # Track action history
        self.last_actions.append(action)
        if len(self.last_actions) > 5:
            self.last_actions.pop(0)
        
        # Measure new performance
        new_select_perf = measure_query_performance(self.cursor)
        new_insert_perf = measure_insert_performance(self.cursor)
        
        self.query_times.append(new_select_perf)
        self.insert_times.append(new_insert_perf)
        
        # Calculate performance changes
        select_improvement = (self.query_times[-2] if len(self.query_times) > 1 else new_select_perf) - new_select_perf
        insert_degradation = new_insert_perf - (self.insert_times[-2] if len(self.insert_times) > 1 else new_insert_perf)
        
        # Update best performance
        if new_select_perf < self.best_performance:
            self.best_performance = new_select_perf
        
        # Calculate reward
        reward = 0
        
        # 1. SELECT performance reward (ΔP)
        if select_improvement > 0:
            reward += select_improvement * 200  # Amplify improvements
            print(f"  📈 SELECT improved: +{select_improvement*1000:.1f}ms")
        elif select_improvement < 0:
            reward += select_improvement * 100  # Penalize degradation
            print(f"  📉 SELECT degraded: {select_improvement*1000:.1f}ms")
        
        # 2. INSERT impact penalty (Coût_A for INSERT slowdown)
        insert_penalty, current_insert_time = self._calculate_insert_impact_penalty(
            get_existing_indexes(self.cursor)
        )
        reward -= insert_penalty
        
        if insert_penalty > 0:
            print(f"  ⚠️ INSERT penalty: -{insert_penalty:.2f} (time: {current_insert_time*1000:.1f}ms)")
        
        # 3. Action cost
        if action == 1 and action_successful:
            reward -= INDEX_CREATION_COST * 10
            print(f"  💰 CREATE cost: -{INDEX_CREATION_COST*10:.2f}")
        elif action == 2 and action_successful:
            reward -= INDEX_DROP_PENALTY * 8
            print(f"  💰 DROP cost: -{INDEX_DROP_PENALTY*8:.2f}")
        
        # 4. Binary matrix optimization reward
        binary_matrix = get_binary_index_matrix(self.cursor)
        total_indexes = np.sum(binary_matrix)
        
        # Reward for optimal index count (2-3 indexes)
        if 2 <= total_indexes <= 3:
            reward += 0.3
            print(f"  🎯 Optimal index count: {total_indexes} indexes")
        elif total_indexes > 4:
            reward -= (total_indexes - 3) * 0.2
            print(f"  ⚠️ Too many indexes: {total_indexes}")
        
        # 5. Bonus for reaching best performance
        if new_select_perf == self.best_performance and select_improvement > 0:
            reward += 0.5
            print(f"  🏆 New best performance!")
        
        # 6. Penalty for blocked actions
        if action_blocked:
            reward -= 0.3
        
        # 7. Penalty for CREATE spamming
        if len(self.last_actions) >= 3:
            if action == 1 and all(a == 1 for a in self.last_actions[-3:]):
                reward -= 0.4
                print(f"  ⚠️ CREATE spam penalty: -0.4")
        
        # 8. CATASTROPHIC PENALTY for 10+ indexes
        new_index_count = get_existing_indexes(self.cursor)
        if new_index_count >= 10:
            catastrophic_penalty = 8.0 + (new_index_count - 10) * 2.0
            reward -= catastrophic_penalty
            print(f"  🔴 CATASTROPHIC PENALTY: -{catastrophic_penalty:.2f}")
            print(f"     {new_index_count} UNNECESSARY INDEXES CREATED!")
        
        # Clip reward
        reward = np.clip(reward, -15, 10)
        
        print(f"  🎁 Total reward: {reward:.4f}")
        
        # Get new state
        new_state = self._get_state()
        
        # Check termination conditions
        terminated = False
        truncated = self.step_count >= EPISODE_LENGTH
        
        # Early termination for catastrophic behavior
        if new_index_count >= 10:
            terminated = True
            reward -= 3.0  # Additional termination penalty
            print(f"\n🔴 EARLY TERMINATION: Catastrophic index creation detected!")
        
        if terminated or truncated:
            print(f"\n⏹️ Episode ended:")
            print(f"   Steps: {self.step_count}")
            print(f"   Final indexes: {new_index_count}/{self.max_indexes}")
            print(f"   Final SELECT: {new_select_perf:.4f}s")
            print(f"   Final INSERT: {new_insert_perf:.4f}s")
            print(f"   Total reward: {reward:.4f}")
        
        return new_state, reward, terminated, truncated, {
            'action_details': action_details,
            'select_perf': new_select_perf,
            'insert_perf': new_insert_perf,
            'index_count': new_index_count
        }
    
    def close(self):
        """Safely close environment"""
        print("\n🔒 Closing enhanced environment...")
        try:
            self._clear_cursor_results()
            
            # Clean up test table
            try:
                self.cursor.execute("DROP TABLE IF EXISTS perf_test_insert")
            except:
                pass
            
            # Clean up excessive indexes (keep only 2-3)
            try:
                self.cursor.execute("SHOW INDEX FROM orders WHERE Key_name LIKE 'idx_orders_%'")
                indexes = self.cursor.fetchall()
                if len(indexes) > 3:
                    print(f"  🧹 Cleaning up {len(indexes) - 3} excessive indexes")
                    for i in range(3, len(indexes)):
                        try:
                            self.cursor.execute(f"DROP INDEX {indexes[i][2]} ON orders")
                        except:
                            pass
            except:
                pass
            
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
                
            print("✅ Environment closed")
        except Exception as e:
            print(f"⚠️ Error during close: {e}")


# Quick test
if __name__ == "__main__":
    print("🧪 Testing Enhanced Environment")
    print("="*60)
    
    env = MySQLIndexEnvEnhanced()
    state, _ = env.reset()
    
    print(f"\nInitial state shape: {state.shape}")
    print(f"State dimension: {env.observation_space.shape[0]}")
    
    # Test a few actions
    for i in range(5):
        print(f"\n--- Action {i+1} ---")
        action = np.random.choice([0, 1, 2])
        state, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    env.close()
    print("\n✅ Enhanced environment test completed")