import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
from stable_baselines3 import DQN
from env_enhanced import MySQLIndexEnvFinal

def run_evaluation(model_path, env, num_steps=15, deterministic=True):
    """Run evaluation and return detailed results"""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_path}")
    print(f"{'='*60}")
    
    # Load model
    try:
        model = DQN.load(model_path)
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None
    
    obs, _ = env.reset()
    results = {
        'rewards': [],
        'actions': [],
        'index_counts': [],
        'query_times': [],
        'step_details': []
    }
    
    print(f"\nRunning {num_steps} steps ({'deterministic' if deterministic else 'stochastic'} mode):")
    print("-" * 75)
    print(f"{'Step':>5s} {'Action':<10s} {'Reward':>10s} {'Indexes':>12s} {'Query Time':>12s}")
    print("-" * 75)
    
    for step in range(num_steps):
        # Get action
        action, _ = model.predict(obs, deterministic=deterministic)
        
        # Convert to integer if needed
        if isinstance(action, np.ndarray):
            action = int(action[0] if action.shape else action.item())
        
        # Take action
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Store results
        results['rewards'].append(reward)
        results['actions'].append(action)
        current_indexes = int(obs[0] * env.max_indexes)
        results['index_counts'].append(current_indexes)
        results['query_times'].append(obs[1])
        results['step_details'].append({
            'step': step + 1,
            'action': action,
            'reward': reward,
            'indexes': current_indexes,
            'query_time': obs[1]
        })
        
        # Print step info
        action_names = ["NOOP", "CREATE", "DROP"]
        print(f"{step+1:5d} {action_names[action]:<10s} {reward:10.4f} "
              f"{current_indexes:3d}/{env.max_indexes:3d} {obs[1]:12.4f}s")
        
        if terminated or truncated:
            print(f"Episode ended at step {step+1}")
            break
    
    return results

def analyze_results(results, env):
    """Analyze and print evaluation results"""
    if not results:
        return
    
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    rewards = results['rewards']
    actions = results['actions']
    index_counts = results['index_counts']
    query_times = results['query_times']
    
    # Basic statistics
    total_reward = sum(rewards)
    avg_reward = np.mean(rewards)
    
    print(f"Total Reward: {total_reward:.4f}")
    print(f"Average Reward per Step: {avg_reward:.4f}")
    print(f"Best Step Reward: {np.max(rewards):.4f}")
    print(f"Worst Step Reward: {np.min(rewards):.4f}")
    
    # Action distribution
    action_counts = {0: 0, 1: 0, 2: 0}
    for action in actions:
        action_counts[action] += 1
    
    total_actions = len(actions)
    print(f"\nAction Distribution:")
    action_names = ["NOOP", "CREATE", "DROP"]
    for action_type in [0, 1, 2]:
        count = action_counts[action_type]
        percentage = count / total_actions * 100 if total_actions > 0 else 0
        print(f"  {action_names[action_type]:6s}: {count:3d} ({percentage:5.1f}%)")
    
    # Performance metrics
    print(f"\nPerformance Metrics:")
    print(f"  Final Indexes: {index_counts[-1]}/{env.max_indexes}")
    print(f"  Max Indexes Used: {np.max(index_counts)}")
    print(f"  Min Indexes Used: {np.min(index_counts)}")
    print(f"  Final Query Time: {query_times[-1]:.4f}s")
    print(f"  Best Query Time: {np.min(query_times):.4f}s")
    print(f"  Average Query Time: {np.mean(query_times):.4f}s")
    
    # Performance improvement
    if len(query_times) > 1:
        improvement = (query_times[0] - query_times[-1]) / query_times[0] * 100
        print(f"  Overall Improvement: {improvement:.1f}%")
    
    # Behavior analysis
    print(f"\nBehavior Analysis:")
    
    # Check index limit respect
    if np.max(index_counts) <= env.max_indexes:
        print(f"  ✓ Respects index limit (max: {np.max(index_counts)}/{env.max_indexes})")
    else:
        print(f"  ✗ Exceeds index limit (max: {np.max(index_counts)}/{env.max_indexes})")
    
    # Check for CREATE-DROP oscillation
    create_drop_pairs = 0
    for i in range(1, len(actions)):
        if (actions[i-1] == 1 and actions[i] == 2) or (actions[i-1] == 2 and actions[i] == 1):
            create_drop_pairs += 1
    
    if create_drop_pairs > len(actions) * 0.3:  # More than 30% oscillation
        print(f"  ⚠  High CREATE-DROP oscillation ({create_drop_pairs} pairs)")
    else:
        print(f"  ✓ Reasonable action pattern")
    
    # Check reward trend
    if total_reward > 0:
        print(f"  ✓ Achieves positive total reward")
    else:
        print(f"  ✗ Has negative total reward")
    
    # Check query time stability
    query_time_std = np.std(query_times)
    if query_time_std < 0.01:  # Less than 10ms variation
        print(f"  ✓ Stable query performance (std: {query_time_std:.4f}s)")
    else:
        print(f"  ⚠  Variable query performance (std: {query_time_std:.4f}s)")
    
    return {
        'total_reward': total_reward,
        'avg_reward': avg_reward,
        'final_indexes': index_counts[-1],
        'final_query_time': query_times[-1],
        'action_distribution': action_counts
    }

if __name__ == "__main__":
    print("ENHANCED EVALUATION OF INDEX OPTIMIZATION AGENT")
    print("=" * 60)
    
    # Create environment
    env = MySQLIndexEnvFinal()
    
    # Models to evaluate
    models_to_evaluate = [
        "index_optimizer_enhanced",
        "index_optimizer_latest", 
        "index_optimizer_interrupted",
        "index_optimizer_dqn_trained",
        "index_optimizer_dqn_latest"
    ]
    
    all_results = {}
    
    # Evaluate each model
    for model_name in models_to_evaluate:
        try:
            results = run_evaluation(model_name, env, num_steps=15, deterministic=True)
            if results:
                analysis = analyze_results(results, env)
                all_results[model_name] = analysis
                
                # Reset environment for next evaluation
                env.close()
                env = MySQLIndexEnvFinal()
        except:
            pass  # Skip models that don't exist
    
    # Compare all models if we have results
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        
        # Create comparison table
        print(f"\n{'Model':30s} {'Total Reward':>12s} {'Avg Reward':>10s} {'Final Index':>11s} {'Query Time':>11s}")
        print("-" * 80)
        
        sorted_models = sorted(all_results.items(), key=lambda x: x[1]['total_reward'], reverse=True)
        
        for model_name, results in sorted_models:
            print(f"{model_name:30s} {results['total_reward']:12.4f} "
                  f"{results['avg_reward']:10.4f} {results['final_indexes']:3d}/5"
                  f"{results['final_query_time']:11.4f}s")
        
        # Best model recommendation
        best_model_name, best_results = sorted_models[0]
        print(f"\n✓ RECOMMENDED MODEL: {best_model_name}")
        print(f"  Total Reward: {best_results['total_reward']:.4f}")
        print(f"  Final Indexes: {best_results['final_indexes']}/5")
        print(f"  Final Query Time: {best_results['final_query_time']:.4f}s")
    
    env.close()
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")