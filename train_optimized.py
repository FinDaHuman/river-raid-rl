"""High-performance Rainbow DQN training for River Raid with GPU + async envs."""
import os, sys, time, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv

from riverraid_rl.env import make_riverraid_env
from riverraid_rl.config import EnvConfig
from riverraid_rl.utils.evaluation import evaluate
from riverraid_rl.agents.better_than_human import (
    BetterThanHumanRainbowAgent,
    BetterThanHumanRainbowConfig,
)


def train_optimized(
    total_steps=10_000_000,
    num_envs=8,
    eval_freq=100_000,
    save_freq=500_000,
    quick_test=False,
    use_attention=False,
    device="cuda",
    device_auto=True,
):
    if device_auto:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if quick_test:
        num_envs = 4
        total_steps = 5000
        eval_freq = 1000000
        save_freq = 5000000

    N_ACTIONS = 6
    EVAL_EPISODES = 20
    LOG_FREQ = 2_000

    total_frames_estimate = total_steps * EnvConfig().frame_skip * num_envs
    beta_frames_val = max(int(total_frames_estimate * 0.1), 250_000)

    cfg = BetterThanHumanRainbowConfig(
        hidden_dim=256,
        batch_size=32,
        buffer_capacity=100_000 if not quick_test else 5_000,
        min_replay_size=20_000 if not quick_test else 500,
        target_update_freq=8_000,
        train_freq=4,
        learning_rate=0.0001,
        gamma=0.997,
        v_min=-10.0,
        v_max=50.0,
        num_atoms=51,
        n_step=5,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=beta_frames_val,
        max_grad_norm=10.0,
        use_noisy_nets=True,
        use_attention=use_attention,
        teacher_warm_start_steps=20_000,
    )

    run_name = f"human-target-{int(time.time())}"
    save_dir = f"checkpoints/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    env_cfg = EnvConfig()
    agent = BetterThanHumanRainbowAgent(env_cfg, cfg, N_ACTIONS, device, total_steps=total_steps)

    print("=" * 70)
    print("HUMAN-TARGET RAINBOW TRAINING - River Raid")
    print(f"  Device: {device.upper()}  |  Envs: {num_envs} sync  |  Steps: {total_steps:,}")
    print(f"  Buffer: {cfg.buffer_capacity:,}  |  Batch: {cfg.batch_size}  |  N-step: {cfg.n_step}")
    print(f"  LR: {cfg.learning_rate}  |  Gamma: {cfg.gamma}^n={cfg.gamma**cfg.n_step:.4f}  |  Hidden: {cfg.hidden_dim}")
    print(f"  NoisyNets: {cfg.use_noisy_nets}  |  Attention: {cfg.use_attention}")
    print(f"  Atoms: {cfg.num_atoms}  |  V-range: [{cfg.v_min}, {cfg.v_max}]")
    print(f"  Reward clip: sign()  |  Params: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    print("=" * 70)

    print(f"Creating {num_envs} vectorized environments...")
    vec_env = SyncVectorEnv([
        lambda idx=i: make_riverraid_env(env_cfg, clip_rewards=True)
        for i in range(num_envs)
    ])

    states, _ = vec_env.reset()
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    episode_count = 0
    best_mean_reward = float("-inf")

    t_start = time.time()
    last_log_step = 0
    last_save_step = 0
    eval_results = []

    for step in range(total_steps):
        actions = agent.act_batch(np.array(states), training=True)
        next_states, rewards, terminated, truncated, infos = vec_env.step(actions)

        for i in range(num_envs):
            agent.memory.push(states[i], actions[i], rewards[i], next_states[i],
                              bool(terminated[i] or truncated[i]), env_id=i)
            episode_rewards[i] += rewards[i]
            episode_lengths[i] += 1
            if terminated[i] or truncated[i]:
                episode_count += 1

        states = next_states

        if step >= cfg.min_replay_size and step % cfg.train_freq == 0:
            metrics = agent.update()
            if metrics and step - last_log_step >= LOG_FREQ:
                elapsed = time.time() - t_start
                sps = (step + 1) / elapsed if elapsed > 0 else 0
                total_frames = (step + 1) * env_cfg.frame_skip * num_envs
                avg_rew = episode_rewards.mean()
                lr = metrics.get('lr', cfg.learning_rate)
                print(
                    f"[{step:>8,}/{total_steps:,}] "
                    f"{100*(step+1)/total_steps:5.1f}% | "
                    f"{sps:6.0f} sps | "
                    f"{total_frames:>9,} frames | "
                    f"ep={episode_count:>5} | "
                    f"loss={metrics['loss']:.3f} | "
                    f"q={metrics['q_value']:.1f} | "
                    f"beta={metrics['beta']:.3f} | "
                    f"lr={lr:.1e}"
                )
                last_log_step = step

        if step > 0 and step % eval_freq == 0:
            agent.q_network.eval()
            result = evaluate(agent, env_cfg, EVAL_EPISODES)
            agent.q_network.train()
            mean_r = result["mean_reward"]
            elapsed = time.time() - t_start
            total_frames = step * env_cfg.frame_skip * num_envs
            pct_human = 100.0 * mean_r / 13_513
            print(f"\n{'='*70}")
            print(f"EVAL at step {step:,} ({elapsed:.0f}s, {total_frames:,} frames)")
            print(f"  Mean: {mean_r:.1f} +/- {result['std_reward']:.1f}")
            print(f"  Min: {result['min_reward']:.1f}  Max: {result['max_reward']:.1f}")
            print(f"  % of Human Expert: {pct_human:.2f}%")
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
                "pct_human": pct_human,
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
            last_save_step = step

    vec_env.close()
    elapsed_total = time.time() - t_start
    total_frames = total_steps * env_cfg.frame_skip * num_envs
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"  Steps: {total_steps:,}  |  Frames: {total_frames:,}")
    print(f"  Time: {elapsed_total:.0f}s ({total_steps/elapsed_total:.0f} sps)")
    print(f"  Episodes: {episode_count}")
    print(f"  Best eval: {best_mean_reward:.1f}")
    print(f"  % Human Expert (13,513): {100*best_mean_reward/13513:.2f}%")
    print(f"{'='*70}")

    agent.q_network.eval()
    final = evaluate(agent, env_cfg, EVAL_EPISODES * 2)
    print(f"\nFinal evaluation ({EVAL_EPISODES*2} episodes):")
    print(f"  Mean: {final['mean_reward']:.1f} +/- {final['std_reward']:.1f}")
    print(f"  Min: {final['min_reward']:.1f}  Max: {final['max_reward']:.1f}")
    print(f"  % Human Expert: {100*final['mean_reward']/13513:.2f}%")

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
                "pct_human": 100 * final["mean_reward"] / 13513,
            },
            "device": device,
            "use_attention": use_attention,
            "use_noisy_nets": cfg.use_noisy_nets,
            "hyperparameters": {
                "num_envs": num_envs,
                "buffer_capacity": cfg.buffer_capacity,
                "batch_size": cfg.batch_size,
                "n_step": cfg.n_step,
                "learning_rate": cfg.learning_rate,
                "gamma": cfg.gamma,
                "hidden_dim": cfg.hidden_dim,
                "num_atoms": cfg.num_atoms,
                "beta_frames": cfg.beta_frames,
            },
        }, f, indent=2)

    print(f"\nResults saved to {save_dir}/")
    print("Done!")
    return final


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Human-target Rainbow training for River Raid")
    parser.add_argument("--steps", type=int, default=10_000_000, help="Total training steps")
    parser.add_argument("--envs", type=int, default=8, help="Number of parallel async environments")
    parser.add_argument("--eval-freq", type=int, default=100_000, help="Evaluation frequency")
    parser.add_argument("--quick-test", action="store_true", help="Run a quick 5000-step test")
    parser.add_argument("--attention", action="store_true", help="Use attention layers in CNN backbone")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()
    train_optimized(
        total_steps=args.steps,
        num_envs=args.envs,
        eval_freq=args.eval_freq,
        quick_test=args.quick_test,
        use_attention=args.attention,
        device="cpu" if args.cpu else "cuda",
    )
