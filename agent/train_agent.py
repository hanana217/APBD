import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env_enhanced import MySQLIndexEnvFinal  # Use the enhanced environment

# Set random seed
np.random.seed(42)

print("=" * 60)
print("TRAINING NEW MODEL WITH ENHANCED ENVIRONMENT")
print("=" * 60)

def make_env():
    env = MySQLIndexEnvFinal()
    env = Monitor(env)
    return env

# Create vectorized environment
env = DummyVecEnv([make_env])

print("\nInitializing DQN model...")

# Create DQN model with optimized parameters
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    buffer_size=10000,
    learning_starts=300,
    batch_size=64,
    gamma=0.99,
    tau=0.001,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=500,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    exploration_fraction=0.4,
)

print("\nStarting training for 3000 timesteps...")
print("This will take approximately 120 episodes")
print("Press Ctrl+C to stop early")

try:
    model.learn(
        total_timesteps=3000,
        log_interval=100
    )
    
    print("\n✓ Training completed!")
    model.save("index_optimizer_enhanced")
    print("✓ Model saved as 'index_optimizer_enhanced'")
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user.")
    model.save("index_optimizer_interrupted")
    print("✓ Model saved as 'index_optimizer_interrupted'")
except Exception as e:
    print(f"\nTraining error: {e}")

# Always save
try:
    model.save("index_optimizer_latest")
    print("✓ Model saved as 'index_optimizer_latest'")
except:
    pass

# Close training environment
env.close()

print("\n" + "=" * 60)
print("RUNNING EVALUATION")
print("=" * 60)

# Create fresh evaluation environment
eval_env = MySQLIndexEnvFinal()

# Load the best model
try:
    model = DQN.load("index_optimizer_enhanced")
    print("Loaded: index_optimizer_enhanced")
except:
    try:
        model = DQN.load("index_optimizer_latest")
        print("Loaded: index_optimizer_latest")
    except:
        print("No model found for evaluation")
        exit(1)

# Run evaluation
obs, _ = eval_env.reset()
total_reward = 0
action_counts = {0: 0, 1: 0, 2: 0}
step_rewards = []
index_counts = []

print("\nRunning 15-step evaluation (deterministic mode):")
for step in range(15):
    # Get action
    action, _ = model.predict(obs, deterministic=True)
    
    # Convert to integer if needed
    if isinstance(action, np.ndarray):
        action = int(action[0] if action.shape else action.item())
    
    obs, reward, terminated, truncated, _ = eval_env.step(action)
    
    total_reward += reward
    action_counts[action] += 1
    step_rewards.append(reward)
    current_indexes = int(obs[0] * eval_env.max_indexes)
    index_counts.append(current_indexes)
    
    action_names = ["NOOP", "CREATE", "DROP"]
    print(f"Step {step+1}: {action_names[action]:6s} | "
          f"Reward: {reward:7.4f} | "
          f"Indexes: {current_indexes}/{eval_env.max_indexes} | "
          f"Query time: {obs[1]:.4f}s")
    
    if terminated or truncated:
        break

# Print summary
print(f"\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(f"Total Reward: {total_reward:.4f}")
print(f"Average Reward per Step: {np.mean(step_rewards):.4f}")
print(f"Best Step Reward: {np.max(step_rewards):.4f}")
print(f"Worst Step Reward: {np.min(step_rewards):.4f}")

total_actions = sum(action_counts.values())
print(f"\nAction Distribution:")
for action_type, count in action_counts.items():
    action_name = ["NOOP", "CREATE", "DROP"][action_type]
    percentage = count / total_actions * 100 if total_actions > 0 else 0
    print(f"  {action_name:6s}: {count:3d} ({percentage:5.1f}%)")

print(f"\nPerformance Metrics:")
print(f"  Final Indexes: {index_counts[-1]}/{eval_env.max_indexes}")
print(f"  Max Indexes Used: {np.max(index_counts)}")
print(f"  Average Query Time: {np.mean([obs[1] for _ in step_rewards]):.4f}s")

# Check if model respects limits
if index_counts[-1] <= eval_env.max_indexes:
    print(f"  ✓ Model respects index limit")
else:
    print(f"  ✗ Model exceeds index limit ({index_counts[-1]} > {eval_env.max_indexes})")

if total_reward > 0:
    print(f"  ✓ Model achieves positive total reward")
else:
    print(f"  ✗ Model has negative total reward")

eval_env.close()

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)