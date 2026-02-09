# train_enhanced.py
"""
Enhanced training script for binary matrix agent
KEYWORD: ENHANCED_TRAINING
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import time
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env_enhanced import MySQLIndexEnvEnhanced
from agent import EnhancedTrainingCallback

def make_enhanced_env():
    """Create enhanced environment"""
    return MySQLIndexEnvEnhanced()

class EnhancedProgressCallback(BaseCallback):
    """Callback for enhanced training progress"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.reward_history = []
    
    def _on_step(self):
        if self.locals['dones'][0]:
            self.episode_count += 1
            
            if self.episode_count % 5 == 0:
                recent_rewards = self.reward_history[-10:] if len(self.reward_history) >= 10 else self.reward_history
                if recent_rewards:
                    avg_reward = np.mean(recent_rewards)
                    print(f"📈 Episode {self.episode_count} - Avg reward (last 10): {avg_reward:.2f}")
        
        return True

def train_enhanced_agent(total_timesteps=5000, learning_rate=0.0003):
    """Train enhanced agent"""
    print("="*70)
    print("🎓 ENHANCED AGENT TRAINING - BINARY MATRIX")
    print("="*70)
    
    # Create environment
    print("\n1️⃣ Creating enhanced environment...")
    env = DummyVecEnv([make_enhanced_env])
    print("   ✅ Environment created")
    
    # Create model with enhanced parameters
    print("\n2️⃣ Initializing enhanced DQN model...")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=30000,          # Larger buffer for complex state
        learning_starts=1000,       # More exploration
        batch_size=128,
        gamma=0.98,                 # Slightly higher for long-term learning
        tau=0.001,                  # Slower target update
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02, # Less final exploration
        exploration_fraction=0.25,
        verbose=1,
        tensorboard_log="./enhanced_tensorboard/"
    )
    
    print("   ✅ Enhanced model initialized")
    print(f"   📊 Parameters:")
    print(f"      - Learning rate: {learning_rate}")
    print(f"      - Buffer size: 30000")
    print(f"      - Total timesteps: {total_timesteps}")
    print(f"      - State dimensions: {env.observation_space.shape}")
    
    # Callbacks
    callback = EnhancedTrainingCallback(check_freq=100)
    
    # Training
    print(f"\n3️⃣ Starting enhanced training ({total_timesteps} timesteps)...")
    print("   ⏱️ Estimated time: 15-25 minutes")
    print("   💡 Press Ctrl+C to stop gracefully\n")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("✅ ENHANCED TRAINING COMPLETE!")
        print("="*70)
        print(f"⏱️ Total time: {training_time/60:.1f} minutes")
        print(f"📊 Episodes completed: {len(callback.episode_rewards)}")
        print(f"🏆 Best mean reward: {callback.best_mean_reward:.2f}")
        
        # Save models
        model.save("index_optimizer_enhanced_final")
        model.save("index_optimizer_enhanced_latest")
        
        print("\n💾 Models saved:")
        print("   - index_optimizer_enhanced_final")
        print("   - index_optimizer_enhanced_latest")
        
        if callback.best_mean_reward > 0:
            print("\n🎯 Training successful! The agent has learned effective strategies.")
        else:
            print("\n⚠️ Training completed but rewards were low. Consider adjusting parameters.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        model.save("index_optimizer_enhanced_interrupted")
        print("💾 Model saved: index_optimizer_enhanced_interrupted")
        training_time = time.time() - start_time
    
    except Exception as e:
        print(f"\n❌ Training error: {e}")
    
    finally:
        env.close()
        print("\n🔒 Environment closed")
    
    return model, callback

def evaluate_enhanced_agent(model_path="index_optimizer_enhanced_latest", num_steps=20):
    """Evaluate enhanced agent"""
    print("\n" + "="*70)
    print("📊 ENHANCED AGENT EVALUATION")
    print("="*70)
    
    try:
        model = DQN.load(model_path)
        print(f"✅ Loaded model: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    env = MySQLIndexEnvEnhanced()
    obs, _ = env.reset()
    
    print(f"\nRunning {num_steps} steps of evaluation:")
    print("-"*80)
    
    results = {
        'rewards': [],
        'actions': [],
        'indexed_counts': [],
        'select_perfs': [],
        'insert_perfs': []
    }
    
    for step in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract binary matrix and count indexed columns
        binary_matrix = obs[:env.total_columns]
        indexed_count = int(np.sum(binary_matrix > 0.5))
        
        results['rewards'].append(reward)
        results['actions'].append(action)
        results['indexed_counts'].append(indexed_count)
        results['select_perfs'].append(info.get('select_perf', 0))
        results['insert_perfs'].append(info.get('insert_perf', 0))
        
        action_names = ["NOOP", "CREATE", "DROP"]
        print(f"Step {step+1:3d}: {action_names[action]:6s} | "
              f"Reward: {reward:7.3f} | "
              f"Indexed: {indexed_count:2d} | "
              f"SELECT: {info.get('select_perf', 0):.4f}s | "
              f"INSERT: {info.get('insert_perf', 0):.4f}s")
        
        if terminated or truncated:
            print(f"Episode ended at step {step+1}")
            break
    
    # Analysis
    print("\n" + "="*70)
    print("📈 EVALUATION RESULTS")
    print("="*70)
    
    total_reward = sum(results['rewards'])
    avg_reward = np.mean(results['rewards'])
    
    print(f"Total Reward: {total_reward:.4f}")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Best Reward: {np.max(results['rewards']):.4f}")
    print(f"Worst Reward: {np.min(results['rewards']):.4f}")
    
    # Action distribution
    action_counts = {0: 0, 1: 0, 2: 0}
    for action in results['actions']:
        action_counts[action] += 1
    
    print(f"\nAction Distribution:")
    action_names = ["NOOP", "CREATE", "DROP"]
    for action_type in [0, 1, 2]:
        count = action_counts[action_type]
        percentage = count / len(results['actions']) * 100
        print(f"  {action_names[action_type]:6s}: {count:3d} ({percentage:5.1f}%)")
    
    # Performance metrics
    print(f"\nPerformance Metrics:")
    print(f"  Final indexed columns: {results['indexed_counts'][-1]}")
    print(f"  Average SELECT: {np.mean(results['select_perfs']):.4f}s")
    print(f"  Average INSERT: {np.mean(results['insert_perfs']):.4f}s")
    
    # Improvement
    if len(results['select_perfs']) > 1:
        improvement = (results['select_perfs'][0] - results['select_perfs'][-1]) / results['select_perfs'][0] * 100
        print(f"  SELECT Improvement: {improvement:+.1f}%")
    
    env.close()
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        # Evaluation mode
        model_path = sys.argv[2] if len(sys.argv) > 2 else "index_optimizer_enhanced_latest"
        num_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 15
        evaluate_enhanced_agent(model_path, num_steps)
    else:
        # Training mode
        timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
        lr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0003
        model, callback = train_enhanced_agent(total_timesteps=timesteps, learning_rate=lr)
        
        # Quick evaluation after training
        print("\n" + "="*70)
        print("🧪 QUICK POST-TRAINING EVALUATION")
        print("="*70)
        evaluate_enhanced_agent("index_optimizer_enhanced_latest", 10)