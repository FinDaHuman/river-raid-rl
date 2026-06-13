"""Human-level River Raid training — optimized for RTX 3050 Ti.

Key improvements over baseline:
- AsyncVectorEnv for parallel CPU env stepping (better GPU saturation)
- Attention mechanism for spatial reasoning (combined with NoisyNets)
- Larger buffer, batch, and hidden dim for learning capacity
- torch.compile for GPU kernel fusion
- Logging to file for crash recovery
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv

from riverraid_rl.env import make_riverraid_env
from riverraid_rl.config import EnvConfig
from riverraid_rl.utils.evaluation import evaluate
from riverraid_rl.agents.better_than_human import (
    BetterThanHumanRainbowAgent,
    BetterThanHumanRainbowConfig,
)


def _make_env():
    return make_riverraid_env(EnvConfig(), clip_rewards=True)


LOG_FILE = "train_output.log"


def make_agent(env_cfg, device, total_steps=5_000_000):
    cfg = BetterThanHumanRainbowConfig(
        hidden_dim=512,
        batch_size=64,
        buffer_capacity=10_000,
        min_replay_size=30_000,
        target_update_freq=8_000,
        train_freq=4,
        learning_rate=0.0001,
        gamma=0.997,
        v_min=-10.0,
        v_max=200.0,
        num_atoms=51,
        n_step=5,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=1_000_000,
        max_grad_norm=10.0,
        use_noisy_nets=True,
        use_attention=True,
        teacher_warm_start_steps=20_000,
    )
    agent = BetterThanHumanRainbowAgent(env_cfg, cfg, 6, device, total_steps=total_steps)
    return agent, cfg


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def train_human_level(total_steps=5_000_000, num_envs=4, eval_freq=500_000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_compile = device == "cuda"
    env_cfg = EnvConfig()
    agent, cfg = make_agent(env_cfg, device, total_steps)

    if use_compile and hasattr(torch, "compile"):
        log("Compiling Q-networks with torch.compile...")
        try:
            agent.q_network = torch.compile(agent.q_network, mode="reduce-overhead")
            agent.target_network = torch.compile(agent.target_network, mode="reduce-overhead")
        except Exception as e:
            log(f"torch.compile failed, using eager mode: {e}")
            use_compile = False

    run_name = f"human-level-{int(time.time())}"
    save_dir = f"checkpoints/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    total_params = sum(p.numel() for p in agent.q_network.parameters())
    header = [
        "=" * 70,
        "  HUMAN-LEVEL RIVER RAID TRAINING (OPTIMIZED)",
        f"  Device: {device.upper()}  |  Envs: {num_envs}  |  Steps: {total_steps:,}",
        f"  Buffer: {cfg.buffer_capacity:,}  |  Batch: {cfg.batch_size}  |  N-step: {cfg.n_step}",
        f"  LR: {cfg.learning_rate}  |  Gamma: {cfg.gamma}  |  Hidden: {cfg.hidden_dim}",
        f"  NoisyNets: {cfg.use_noisy_nets}  |  Attention: {cfg.use_attention}  |  Compile: {use_compile}",
        f"  Atoms: {cfg.num_atoms}  |  V-range: [{cfg.v_min}, {cfg.v_max}]",
        f"  Params: {total_params:,}  |  Eval freq: {eval_freq:,}",
        f"  Save dir: {save_dir}",
        "=" * 70,
    ]
    for line in header:
        log(line)

    log(f"Creating {num_envs} async vectorized environments...")
    try:
        vec_env = AsyncVectorEnv([_make_env for _ in range(num_envs)])
    except Exception:
        log("AsyncVectorEnv failed, falling back to SyncVectorEnv")
        from gymnasium.vector import SyncVectorEnv
        vec_env = SyncVectorEnv([_make_env for _ in range(num_envs)])

    states, _ = vec_env.reset()
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    episode_count = 0
    best_mean_reward = float("-inf")
    eval_results = []
    t_start = time.time()
    last_log_step = 0
    LOG_FREQ = 5_000
    next_eval = eval_freq
    next_save = 2_000_000
    metrics = None

    log(f"\n{'='*70}")
    log(f"  Step{'':>8}  Progress  |  SPS  |  Frames{'':>6}  |  Ep  |  Loss  |  Q-val  |  Beta  |  LR")
    log(f"{'='*70}")

    for step in range(total_steps):
        actions = agent.act_batch(np.array(states), training=True)
        next_states, rewards, terminated, truncated, _ = vec_env.step(actions)

        for i in range(num_envs):
            agent.memory.push(states[i], actions[i], rewards[i], next_states[i],
                              bool(terminated[i] or truncated[i]), env_id=i)
            episode_rewards[i] += rewards[i]
            if terminated[i] or truncated[i]:
                episode_count += 1
                episode_rewards[i] = 0.0

        states = next_states

        if step >= cfg.min_replay_size and step % cfg.train_freq == 0:
            metrics = agent.update()

        if metrics and step - last_log_step >= LOG_FREQ:
            elapsed = time.time() - t_start
            sps = (step + 1) / elapsed
            total_frames = (step + 1) * env_cfg.frame_skip * num_envs
            lr = metrics.get("lr", cfg.learning_rate)
            log(
                f"  [{step:>8,}/{total_steps:,}] "
                f"{100*(step+1)/total_steps:5.1f}% | "
                f"{sps:6.0f} sps | "
                f"{total_frames:>9,} | "
                f"ep={episode_count:>5} | "
                f"loss={metrics['loss']:.3f} | "
                f"q={metrics['q_value']:.1f} | "
                f"beta={metrics['beta']:.3f} | "
                f"lr={lr:.1e}"
            )
            last_log_step = step

        if step > 0 and step >= next_eval:
            elapsed = time.time() - t_start
            agent.q_network.eval()
            result = evaluate(agent, env_cfg, 20)
            agent.q_network.train()
            mean_r = result["mean_reward"]
            total_frames = step * env_cfg.frame_skip * num_envs
            pct_human = 100.0 * mean_r / 13_513
            eval_lines = [
                f"\n{'='*70}",
                f"  EVAL step={step:,}  elapsed={format_time(elapsed)}  frames={total_frames:,}",
                f"  Mean: {mean_r:.1f} +/- {result['std_reward']:.1f}",
                f"  Min: {result['min_reward']:.1f}  Max: {result['max_reward']:.1f}",
                f"  % Human Expert: {pct_human:.3f}%",
            ]
            if mean_r > best_mean_reward:
                best_mean_reward = mean_r
                agent.save(f"{save_dir}/best.pt")
                eval_lines.append(f"  *** NEW BEST: {best_mean_reward:.1f} ***")
            eval_lines.append(f"{'='*70}\n")
            for line in eval_lines:
                log(line)

            eval_results.append({
                "step": int(step), "elapsed": elapsed,
                "mean_reward": mean_r, "std_reward": result["std_reward"],
                "min_reward": result["min_reward"], "max_reward": result["max_reward"],
                "pct_human": pct_human, "best": best_mean_reward,
                "frames": total_frames,
            })
            with open(f"{save_dir}/progress.json", "w") as f:
                json.dump({
                    "step": step, "best_mean_reward": best_mean_reward,
                    "episodes": episode_count, "elapsed": elapsed,
                    "eval_results": eval_results,
                }, f, indent=2)
            next_eval = step + eval_freq
            last_log_step = step

        if step > 0 and step >= next_save:
            agent.save(f"{save_dir}/step_{step}.pt")
            log(f"  Checkpoint saved at step {step:,}")
            next_save = step + 2_000_000

    vec_env.close()
    elapsed_total = time.time() - t_start
    final_step = step

    log(f"\n{'='*70}")
    log(f"  TRAINING COMPLETE")
    log(f"  Steps: {final_step:,}  |  Time: {format_time(elapsed_total)}")
    log(f"  Env steps: {final_step * num_envs:,}")
    log(f"  Frames: {final_step * env_cfg.frame_skip * num_envs:,}")
    log(f"  Episodes: {episode_count}  |  Best eval: {best_mean_reward:.1f}")
    log(f"  % Human Expert: {100*best_mean_reward/13513:.3f}%")
    log(f"{'='*70}\n")

    agent.q_network.eval()
    final = evaluate(agent, env_cfg, 20)
    log(f"Final evaluation (20 episodes):")
    log(f"  Mean: {final['mean_reward']:.1f} +/- {final['std_reward']:.1f}")
    log(f"  Min: {final['min_reward']:.1f}  Max: {final['max_reward']:.1f}")
    log(f"  % Human Expert: {100*final['mean_reward']/13513:.3f}%")

    agent.save(f"{save_dir}/final.pt")
    with open(f"{save_dir}/summary.json", "w") as f:
        json.dump({
            "total_steps": final_step,
            "total_env_steps": final_step * num_envs,
            "total_frames": final_step * env_cfg.frame_skip * num_envs,
            "elapsed_seconds": elapsed_total,
            "steps_per_second": final_step / elapsed_total,
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
            "compile": use_compile,
            "hyperparameters": cfg.__dict__,
        }, f, indent=2)

    log(f"\nResults saved to {save_dir}/")
    return final


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--eval-freq", type=int, default=500_000)
    args = parser.parse_args()
    train_human_level(total_steps=args.steps, num_envs=args.envs, eval_freq=args.eval_freq)
