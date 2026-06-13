"""CPU-optimized training for Rainbow on Riverraid."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"D:\Vs Code\RELproject")

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import numpy as np
from riverraid_rl.env import make_riverraid_env
from riverraid_rl.config import EnvConfig, RainbowConfig
from riverraid_rl.utils.evaluation import evaluate
from riverraid_rl.agents.rainbow import RainbowAgent

env_cfg = EnvConfig()

# CPU-optimized config
rb_cfg = RainbowConfig()
rb_cfg.hidden_dim = 256
rb_cfg.num_atoms = 21
rb_cfg.v_min = -5
rb_cfg.v_max = 15
rb_cfg.min_replay_size = 5000
rb_cfg.buffer_capacity = 50000
rb_cfg.batch_size = 32
rb_cfg.target_update_freq = 2000
rb_cfg.train_freq = 4
rb_cfg.learning_rate = 0.00025
rb_cfg.epsilon_decay_steps = 50000
rb_cfg.beta_frames = 50000

TOTAL_STEPS = 200000
EVAL_FREQ = 25000
LOG_FREQ = 2000
SAVE_FREQ = 50000

print("=" * 60)
print("RIVERRAID RL - CPU TRAINING (Rainbow)")
print(f"Steps: {TOTAL_STEPS}, Hidden: {rb_cfg.hidden_dim}, Atoms: {rb_cfg.num_atoms}")
print(f"Buffer: {rb_cfg.buffer_capacity}, MinReplay: {rb_cfg.min_replay_size}")
print("=" * 60)

agent = RainbowAgent(env_cfg, rb_cfg, 6, "cpu")
env = make_riverraid_env(env_cfg)

state, info = env.reset()
episode_num = 0
best_mean_reward = float("-inf")
t_start = time.time()

for step in range(TOTAL_STEPS):
    action = agent.act(np.array(state), training=True)
    next_state, reward, terminated, truncated, info = env.step(action)
    agent.memory.push(state, action, reward, next_state, terminated or truncated)
    state = next_state

    if step >= rb_cfg.min_replay_size and step % rb_cfg.train_freq == 0:
        metrics = agent.update()
        if metrics and step % LOG_FREQ == 0:
            elapsed = time.time() - t_start
            sps = step / elapsed if elapsed > 0 else 0
            print(f"[Step {step:>6}] [{elapsed:6.0f}s] loss={metrics['loss']:.3f} q={metrics['q_value']:.1f} eps={metrics['epsilon']:.3f} beta={metrics['beta']:.3f} | ep={episode_num} | {sps:.0f} sps")

    if terminated or truncated:
        state, info = env.reset()
        episode_num += 1

    if step > 0 and step % EVAL_FREQ == 0:
        result = evaluate(agent, env_cfg, 5)
        mean = result["mean_reward"]
        print(f"\n*** EVAL at step {step}: mean={mean:.1f} +/- {result['std_reward']:.1f} len={result['mean_length']:.1f}")
        if mean > best_mean_reward:
            best_mean_reward = mean
            os.makedirs("checkpoints/rainbow-cpu", exist_ok=True)
            agent.save("checkpoints/rainbow-cpu/best.pt")
            print(f"*** New best!")
        print()

    if step > 0 and step % SAVE_FREQ == 0:
        os.makedirs("checkpoints/rainbow-cpu", exist_ok=True)
        agent.save(f"checkpoints/rainbow-cpu/step_{step}.pt")

env.close()
elapsed_total = time.time() - t_start
print(f"\n{'=' * 60}")
print(f"TRAINING COMPLETE: {TOTAL_STEPS} steps in {elapsed_total:.0f}s ({TOTAL_STEPS/elapsed_total:.0f} steps/s)")
print(f"Episodes: {episode_num}, Best eval: {best_mean_reward:.1f}")
print(f"{'=' * 60}")

# Final evaluation
final_result = evaluate(agent, env_cfg, 10)
print(f"\nFinal evaluation (10 episodes):")
print(f"  Mean reward: {final_result['mean_reward']:.1f} +/- {final_result['std_reward']:.1f}")
print(f"  Min: {final_result['min_reward']:.1f}  Max: {final_result['max_reward']:.1f}")
print(f"  Mean length: {final_result['mean_length']:.1f}")
agent.save("checkpoints/rainbow-cpu/final.pt")
print("Final checkpoint saved.")
