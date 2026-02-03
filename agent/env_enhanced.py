import gymnasium as gym
import numpy as np
from gymnasium import spaces
from mysql_utils import *
from config import *
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class MySQLIndexEnvFinal(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Actions: 0=NOOP, 1=CREATE, 2=DROP
        self.action_space = spaces.Discrete(3)
        
        # State: [num_indexes (normalized), last_query_time, query_time_change]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -0.5], dtype=np.float32),
            high=np.array([1.0, 1.0, 0.5], dtype=np.float32),
            dtype=np.float32
        )
        
        self.conn = get_connection()
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()
        self.query_times = []
        self.max_indexes = MAX_INDEXES
        self.step_count = 0
        self.best_performance = None
        self.index_history = []
        self.last_actions = []  # Track last actions to prevent oscillation
        
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
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        self.query_times = []
        self.index_history = []
        self.last_actions = []
        self.best_performance = None
        self._clear_cursor_results()
        
        # Clean up all test indexes
        try:
            self.cursor.execute("SHOW INDEX FROM orders")
            indexes = self.cursor.fetchall()
            for idx in indexes:
                if idx[2].startswith('idx_orders'):
                    try:
                        self.cursor.execute(f"DROP INDEX {idx[2]} ON orders")
                        self._clear_cursor_results()
                    except:
                        pass
        except Exception as e:
            print(f"Reset cleanup error: {e}")
        
        # Measure initial performance
        self.prev_perf = measure_query_performance(self.cursor)
        self.best_performance = self.prev_perf
        self.query_times.append(self.prev_perf)
        current_indexes = get_existing_indexes(self.cursor)
        self.index_history.append(current_indexes)
        
        # Return state
        state = np.array([
            current_indexes / self.max_indexes,
            self.prev_perf,
            0.0  # No change yet
        ], dtype=np.float32)
        
        return state, {}
    
    def step(self, action):
        self.step_count += 1
        current_indexes = get_existing_indexes(self.cursor)
        cost = 0
        action_successful = False
        action_blocked = False
        
        # Check action feasibility
        if action == 1 and current_indexes >= self.max_indexes:  # CREATE when at max
            action_blocked = True
            cost = 0.5  # Penalty for trying to exceed limit
            print(f"[CREATE BLOCKED] Already at max indexes ({current_indexes}/{self.max_indexes})")
        elif action == 2 and current_indexes <= 0:  # DROP when no indexes
            action_blocked = True
            cost = 0.1
            print(f"[DROP BLOCKED] No indexes to drop")
        
        if not action_blocked:
            # Generate unique index name
            timestamp = int(time.time() * 1000)
            
            if action == 1:  # CREATE
                try:
                    index_name = f"idx_orders_{timestamp}"
                    self.cursor.execute(f"""
                        CREATE INDEX {index_name} ON orders(client_id, orderdate)
                    """)
                    self._clear_cursor_results()
                    cost = INDEX_CREATION_COST
                    action_successful = True
                    print(f"[CREATE] Index: {index_name}")
                except Exception as e:
                    print(f"[CREATE Failed] {e}")
                    cost = 0.05  # Cost for failed attempt
                    
            elif action == 2:  # DROP
                try:
                    # Find an index to drop
                    self.cursor.execute("SHOW INDEX FROM orders WHERE Key_name LIKE 'idx_orders_%'")
                    indexes = self.cursor.fetchall()
                    if indexes:
                        index_to_drop = indexes[0][2]  # Get first index name
                        self.cursor.execute(f"DROP INDEX {index_to_drop} ON orders")
                        self._clear_cursor_results()
                        cost = INDEX_DROP_PENALTY
                        action_successful = True
                        print(f"[DROP] Index: {index_to_drop}")
                except Exception as e:
                    print(f"[DROP Failed] {e}")
                    cost = 0.05
            else:  # NOOP
                cost = 0
        
        # Track last action
        self.last_actions.append(action)
        if len(self.last_actions) > 5:
            self.last_actions.pop(0)
        
        # Measure new performance
        new_perf = measure_query_performance(self.cursor)
        
        # Calculate performance improvement
        performance_improvement = (self.prev_perf - new_perf)  # Positive = improvement
        
        # Update best performance
        if new_perf < self.best_performance:
            self.best_performance = new_perf
        
        # Calculate query_time_change (normalized)
        query_time_change = performance_improvement / (self.prev_perf + 1e-6)  # Avoid division by zero
        
        # Calculate reward components
        reward = 0
        
        # 1. Performance improvement reward (main component)
        if performance_improvement > 0:
            reward += performance_improvement * 200  # Large reward for improvement
        else:
            reward += performance_improvement * 100  # Smaller penalty for degradation
        
        # 2. Action cost penalty
        reward -= cost * 10
        
        # 3. Index count optimization
        normalized_indexes = current_indexes / self.max_indexes
        target_indexes = 0.3  # Aim for 30% of max (balanced approach)
        index_distance = abs(normalized_indexes - target_indexes)
        reward -= index_distance * 0.3  # Penalty for being far from target
        
        # 4. Heavy penalty for exceeding max indexes (after action)
        new_index_count = get_existing_indexes(self.cursor)
        if new_index_count > self.max_indexes:
            overflow = new_index_count - self.max_indexes
            reward -= overflow * 0.5 * 10
            print(f"[OVERFLOW PENALTY] {overflow} indexes over limit!")
        
        # 5. Bonus for reaching new best performance
        if new_perf == self.best_performance and performance_improvement > 0:
            reward += 0.5
        
        # 6. Penalty for blocked actions (trying invalid actions)
        if action_blocked:
            reward -= 0.2
        
        # 7. Small penalty for too many consecutive same actions (FIXED)
        if len(self.last_actions) >= 3:
            # Check if last 3 actions are all CREATE (action=1)
            if action == 1 and all(a == 1 for a in self.last_actions[-3:]):
                reward -= 0.1  # Penalty for creating too many in a row
            # Check if last 3 actions are all DROP (action=2)
            elif action == 2 and all(a == 2 for a in self.last_actions[-3:]):
                reward -= 0.1  # Penalty for dropping too many in a row
            # Penalty for CREATE-DROP oscillation
            if len(self.last_actions) >= 4:
                if self.last_actions[-4:] == [1, 2, 1, 2] or self.last_actions[-4:] == [2, 1, 2, 1]:
                    reward -= 0.2
        
        # 8. Bonus for maintaining good performance
        if new_perf < 0.04 and performance_improvement >= 0:  # Good performance threshold
            reward += 0.1
        
        # Clip reward to reasonable range
        reward = np.clip(reward, -5, 5)
        
        print(f"Step {self.step_count}: Action={['NOOP', 'CREATE', 'DROP'][action]}, "
              f"Perf: {self.prev_perf:.4f}s→{new_perf:.4f}s ({performance_improvement*100:.1f}%), "
              f"Indexes: {current_indexes}→{new_index_count}/{self.max_indexes}, "
              f"Reward: {reward:.4f}")
        
        self.prev_perf = new_perf
        self.query_times.append(new_perf)
        self.index_history.append(new_index_count)
        
        # Return state
        state = np.array([
            new_index_count / self.max_indexes,
            new_perf,
            query_time_change
        ], dtype=np.float32)
        
        terminated = False
        truncated = self.step_count >= EPISODE_LENGTH
        
        return state, reward, terminated, truncated, {}
    
    def close(self):
        """Safely close the environment"""
        try:
            self._clear_cursor_results()
            
            # Optional: Keep indexes for analysis
            # Uncomment to clean up all indexes
            # try:
            #     self.cursor.execute("SHOW INDEX FROM orders")
            #     indexes = self.cursor.fetchall()
            #     for idx in indexes:
            #         if idx[2].startswith('idx_orders'):
            #             try:
            #                 self.cursor.execute(f"DROP INDEX {idx[2]} ON orders")
            #                 self._clear_cursor_results()
            #             except:
            #                 pass
            # except:
            #     pass
            
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
        except Exception as e:
            print(f"Warning: Error during close: {e}")