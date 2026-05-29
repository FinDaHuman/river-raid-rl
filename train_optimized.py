"""High-performance Rainbow DQN training for River Raid with vectorized envs."""
import os, sys, time, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from gymnasium.vector import SyncVectorEnv

from riverraid_rl.env import make_riverraid_env
from riverraid_rl.config import EnvConfig, RainbowConfig
from riverraid_rl.utils.evaluation import evaluate
from riverraid_rl.agents.rainbow import RainbowAgent


class OptimizedRainbowAgent(RainbowAgent):
    def act_batch(self, states: np.ndarray, training: bool = True) -> np.ndarray:
        state_t = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            dist = self.q_network(state_t)
            q_values = (dist * self.atoms).sum(dim=2)
        actions = q_values.argmax(dim=1).cpu().numpy()
        if training:
            explore = np.random.random(states.shape[0]) < self.epsilon
            actions[explore] = np.random.randint(0, self.num_actions, size=explore.sum())
        return actions


class RatesSchedule:
    def __init__(self, start, end, decay_steps):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps

    def __call__(self, step):
        frac = min(step / self.decay_steps, 1.0)
        return self.start + (self.end - self.start) * frac


def train_optimized(
    total_steps=5_000_000,
    num_envs=8,
    eval_freq=100_000,
    save_freq=500_000,
    quick_test=False,
):
    if quick_test:
        num_envs = 4
        total_steps = 2000
        eval_freq = 1000000
        save_freq = 5000000

    N_ACTIONS = 6
    EVAL_EPISODES = 20
    LOG_FREQ = 5_000
    TRAIN_FREQ = 4
    TARGET_UPDATE_FREQ = 2_000
    MIN_REPLAY_SIZE = 50_000 if not quick_test else 200
    BATCH_SIZE = 64 if not quick_test else 16
    BUFFER_CAPACITY = 500_000 if not quick_test else 5_000
    LR = 0.0001
    GAMMA = 0.99
    EPSILON_DECAY = 1_000_000
    EPSILON_END = 0.01
    HIDDEN_DIM = 512 if not quick_test else 128
    NUM_ATOMS = 51 if not quick_test else 11
    V_MIN = -10 if not quick_test else -5
    V_MAX = 50 if not quick_test else 15
    N_STEP = 5 if not quick_test else 3
    MAX_GRAD_NORM = 10.0
    BETA_FRAMES = 500_000

    run_name = f"rainbow-opt-{int(time.time())}"
    save_dir = f"checkpoints/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    env_cfg = EnvConfig()
    rb_cfg = RainbowConfig()
    rb_cfg.hidden_dim = HIDDEN_DIM
    rb_cfg.num_atoms = NUM_ATOMS
    rb_cfg.v_min = V_MIN
    rb_cfg.v_max = V_MAX
    rb_cfg.min_replay_size = MIN_REPLAY_SIZE
    rb_cfg.buffer_capacity = BUFFER_CAPACITY
    rb_cfg.batch_size = BATCH_SIZE
    rb_cfg.target_update_freq = TARGET_UPDATE_FREQ
    rb_cfg.train_freq = TRAIN_FREQ
    rb_cfg.learning_rate = LR
    rb_cfg.gamma = GAMMA
    rb_cfg.epsilon_decay_steps = EPSILON_DECAY
    rb_cfg.epsilon_end = EPSILON_END
    rb_cfg.beta_frames = BETA_FRAMES
    rb_cfg.max_grad_norm = MAX_GRAD_NORM
    rb_cfg.n_step = N_STEP

    torch.set_num_threads(8)

    agent = OptimizedRainbowAgent(env_cfg, rb_cfg, N_ACTIONS, "cpu")

    print("=" * 70)
    print("OPTIMIZED RAINBOW TRAINING - River Raid")
    print(f"  Envs: {num_envs} parallel  |  Steps: {total_steps:,}")
    print(f"  Buffer: {BUFFER_CAPACITY:,}  |  Batch: {BATCH_SIZE}  |  N-step: {N_STEP}")
    print(f"  LR: {LR}  |  Gamma: {GAMMA}  |  Hidden: {HIDDEN_DIM}")
    print(f"  Epsilon decay: {EPSILON_DECAY:,} steps  |  Atoms: {NUM_ATOMS}")
    print(f"  Params: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    print("=" * 70)

    print("Creating vectorized environments...")
    vec_env = SyncVectorEnv([
        lambda idx=i: make_riverraid_env(env_cfg, clip_rewards=False)
        for i in range(num_envs)
    ])

    states, _ = vec_env.reset()
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    episode_count = 0
    best_mean_reward = float("-inf")

    t_start = time.time()
    last_log_step = 0
    eval_results = []

    for step in range(total_steps):
        actions = agent.act_batch(np.array(states), training=True)
        next_states, rewards, terminated, truncated, infos = vec_env.step(actions)

        for i in range(num_envs):
            agent.memory.push(states[i], actions[i], rewards[i], next_states[i],
                              bool(terminated[i] or truncated[i]))
            episode_rewards[i] += rewards[i]
            episode_lengths[i] += 1
            if terminated[i] or truncated[i]:
                episode_count += 1

        states = next_states

        if step >= MIN_REPLAY_SIZE and step % TRAIN_FREQ == 0:
            metrics = agent.update()
            if metrics and step - last_log_step >= LOG_FREQ:
                elapsed = time.time() - t_start
                sps = (step + 1) / elapsed if elapsed > 0 else 0
                total_frames = (step + 1) * env_cfg.frame_skip
                eps_count = episode_count if episode_count > 0 else 1
                avg_rew = episode_rewards.mean()
                print(
                    f"[{step:>7,}/{total_steps:,}] "
                    f"{100*(step+1)/total_steps:5.1f}% | "
                    f"{sps:5.0f} sps | "
                    f"{total_frames:,} frames | "
                    f"ep={episode_count} | "
                    f"loss={metrics['loss']:.3f} | "
                    f"q={metrics['q_value']:.1f} | "
                    f"eps={metrics['epsilon']:.3f} | "
                    f"beta={metrics['beta']:.3f}"
                )
                last_log_step = step

        if step > 0 and step % eval_freq == 0:
            agent.q_network.eval()
            result = evaluate(agent, env_cfg, EVAL_EPISODES)
            agent.q_network.train()
            mean_r = result["mean_reward"]
            elapsed = time.time() - t_start
            total_frames = step * env_cfg.frame_skip * num_envs
            print(f"\n{'='*70}")
            print(f"EVAL at step {step:,} ({elapsed:.0f}s, {total_frames:,} frames)")
            print(f"  Mean: {mean_r:.1f} +/- {result['std_reward']:.1f}")
            print(f"  Min: {result['min_reward']:.1f}  Max: {result['max_reward']:.1f}")
            print(f"  Mean length: {result['mean_length']:.1f}")
            if mean_r > best_mean_reward:
                best_mean_reward = mean_r
                agent.save(f"{save_dir}/best.pt")
                print(f"  *** NEW BEST: {best_mean_reward:.1f} ***")
            print(f"{'='*70}\n")

            eval_results.append({
                "step": step,
                "mean_reward": mean_r,
                "std_reward": result["std_reward"],
                "min_reward": result["min_reward"],
                "max_reward": result["max_reward"],
                "mean_length": result["mean_length"],
            })

        if step > 0 and step % save_freq == 0:
            agent.save(f"{save_dir}/step_{step}.pt")
            with open(f"{save_dir}/metrics.json", "w") as f:
                json.dump({
                    "eval_results": eval_results,
                    "total_steps": total_steps,
                    "episodes": episode_count,
                    "best_mean_reward": best_mean_reward,
                }, f, indent=2)

    vec_env.close()
    elapsed_total = time.time() - t_start
    total_frames = total_steps * env_cfg.frame_skip * num_envs
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"  Steps: {total_steps:,}  |  Frames: {total_frames:,}")
    print(f"  Time: {elapsed_total:.0f}s ({total_steps/elapsed_total:.0f} sps)")
    print(f"  Episodes: {episode_count}")
    print(f"  Best eval: {best_mean_reward:.1f}")
    print(f"{'='*70}")

    agent.q_network.eval()
    final = evaluate(agent, env_cfg, EVAL_EPISODES * 2)
    print(f"\nFinal evaluation ({EVAL_EPISODES*2} episodes):")
    print(f"  Mean: {final['mean_reward']:.1f} +/- {final['std_reward']:.1f}")
    print(f"  Min: {final['min_reward']:.1f}  Max: {final['max_reward']:.1f}")

    agent.save(f"{save_dir}/final.pt")
    with open(f"{save_dir}/summary.json", "w") as f:
        json.dump({
            "total_steps": total_steps,
            "total_frames": total_frames,
            "elapsed_seconds": elapsed_total,
            "steps_per_second": total_steps / elapsed_total,
            "episodes": episode_count,
            "best_mean_reward": best_mean_reward,
            "final_eval": {
                "mean_reward": final["mean_reward"],
                "std_reward": final["std_reward"],
                "min_reward": final["min_reward"],
                "max_reward": final["max_reward"],
            },
            "hyperparameters": {
                "num_envs": num_envs,
                "buffer_capacity": BUFFER_CAPACITY,
                "batch_size": BATCH_SIZE,
                "n_step": N_STEP,
                "learning_rate": LR,
                "gamma": GAMMA,
                "hidden_dim": HIDDEN_DIM,
                "num_atoms": NUM_ATOMS,
                "epsilon_decay_steps": EPSILON_DECAY,
            },
        }, f, indent=2)

    print(f"\nResults saved to {save_dir}/")
    print("Done!")
    return final


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimized Rainbow training for River Raid")
    parser.add_argument("--steps", type=int, default=5_000_000, help="Total training steps")
    parser.add_argument("--envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--eval-freq", type=int, default=100_000, help="Evaluation frequency")
    parser.add_argument("--quick-test", action="store_true", help="Run a quick 2000-step test")
    args = parser.parse_args()
    train_optimized(
        total_steps=args.steps,
        num_envs=args.envs,
        eval_freq=args.eval_freq,
        quick_test=args.quick_test,
    )
