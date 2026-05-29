import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import numpy as np

from riverraid_rl.env import make_riverraid_env
from riverraid_rl.config import EnvConfig, RainbowConfig
from riverraid_rl.utils.evaluation import evaluate
from riverraid_rl.utils.progress import ProgressTracker, load_progress
from riverraid_rl.agents.rainbow import RainbowAgent
from riverraid_rl.models.noisy import NoisyLinear


def get_config(quick_test=False, big_model=False):
    env_cfg = EnvConfig()

    rb_cfg = RainbowConfig()
    rb_cfg.hidden_dim = 256 if not big_model else 512
    rb_cfg.num_atoms = 51
    rb_cfg.v_min = -5
    rb_cfg.v_max = 50
    rb_cfg.min_replay_size = 5000 if not quick_test else 500
    rb_cfg.buffer_capacity = 200_000 if not quick_test else 10_000
    rb_cfg.batch_size = 32
    rb_cfg.target_update_freq = 2500
    rb_cfg.train_freq = 4
    rb_cfg.learning_rate = 0.00025
    rb_cfg.gamma = 0.99
    rb_cfg.epsilon_decay_steps = 250_000 if not quick_test else 10_000
    rb_cfg.beta_frames = 250_000 if not quick_test else 10_000
    rb_cfg.n_step = 3

    return env_cfg, rb_cfg


def train_riverraid(
    total_steps=1_000_000,
    eval_freq=50_000,
    eval_episodes=10,
    save_freq=100_000,
    run_name=None,
    quick_test=False,
    big_model=False,
    resume=None,
    no_clip=True,
):
    env_cfg, rb_cfg = get_config(quick_test, big_model)

    if quick_test:
        total_steps = min(total_steps, 10_000)
        eval_freq = 5_000
    if run_name is None:
        run_name = f"rainbow-{total_steps // 1000000}m-{int(time.time())}"

    save_dir = f"checkpoints/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    agent = RainbowAgent(env_cfg, rb_cfg, 6, "cpu")
    start_step = 0
    best_mean_reward = float("-inf")

    if resume and os.path.exists(resume):
        agent.load(resume)
        start_step = agent.steps
        print(f"Resumed from {resume} (step {start_step})")

    tracker = ProgressTracker(save_dir)
    old_progress = load_progress(save_dir)
    if old_progress and old_progress.get("best_mean_reward"):
        best_mean_reward = old_progress["best_mean_reward"]

    total_params = sum(p.numel() for p in agent.q_network.parameters())
    print("=" * 70)
    print("  RIVER RAID RL - UNIFIED TRAINING")
    print(f"  Agent: Rainbow  |  Steps: {total_steps:,}")
    print(f"  Params: {total_params:,}  |  Hidden: {rb_cfg.hidden_dim}  |  Atoms: {rb_cfg.num_atoms}")
    print(f"  Buffer: {rb_cfg.buffer_capacity:,}  |  N-step: {rb_cfg.n_step}  |  V-range: [{rb_cfg.v_min}, {rb_cfg.v_max}]")
    print(f"  LR: {rb_cfg.learning_rate}  |  Gamma: {rb_cfg.gamma}  |  Epsilon decay: {rb_cfg.epsilon_decay_steps:,}")
    print(f"  Eval freq: {eval_freq:,}  |  Eval episodes: {eval_episodes}")
    print(f"  Save dir: {save_dir}")
    print(f"  {'No reward clipping' if no_clip else 'With reward clipping'}")
    if start_step > 0:
        print(f"  Resumed at step {start_step} (best: {best_mean_reward:.1f})")
    print("=" * 70)

    env = make_riverraid_env(env_cfg, clip_rewards=not no_clip)
    state, info = env.reset()
    episode_reward = 0.0
    episode_num = 0
    t_start = time.time()
    last_log = 0
    metrics = None

    ProgressTracker.print_summary_header()

    for step in range(start_step, total_steps):
        action = agent.act(np.array(state), training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.memory.push(state, action, reward, next_state, terminated or truncated)
        state = next_state
        episode_reward += reward

        if step >= rb_cfg.min_replay_size and step % rb_cfg.train_freq == 0:
            metrics = agent.update()

        if metrics and step % 2000 == 0 and step > last_log:
            elapsed = time.time() - t_start
            sps = (step - start_step + 1) / elapsed if elapsed > 0 else 0
            eta = (total_steps - step) / sps if sps > 0 else 0
            print(f"  [{step:>8,}/{total_steps:,}] "
                  f"{100*(step+1)/total_steps:5.1f}% "
                  f"| {sps:5.0f} sps | ETA {eta:6.0f}s "
                  f"| loss={metrics['loss']:.3f} q={metrics['q_value']:.1f} eps={metrics['epsilon']:.3f}",
                  flush=True)
            last_log = step

        if terminated or truncated:
            state, info = env.reset()
            episode_reward = 0.0
            episode_num += 1

        if step > 0 and step % eval_freq == 0:
            elapsed = time.time() - t_start
            total_frames = step * env_cfg.frame_skip

            agent.q_network.eval()
            result = evaluate(agent, env_cfg, eval_episodes)
            agent.q_network.train()

            mean = result["mean_reward"]
            if mean > best_mean_reward:
                best_mean_reward = mean
                agent.save(f"{save_dir}/best.pt")

            row = ProgressTracker.format_eval_row(
                step, total_frames, mean, result["std_reward"],
                100*mean/ProgressTracker.THEORETICAL_MAX,
                100*mean/ProgressTracker.HUMAN_EXPERT,
                100*mean/ProgressTracker.RAINBOW_SOTA,
                result["mean_length"], best_mean_reward, elapsed,
            )
            print(row, flush=True)

            tracker.record_eval(
                step=step, frames=total_frames, episode=episode_num,
                mean_reward=mean, std_reward=result["std_reward"],
                min_reward=result["min_reward"], max_reward=result["max_reward"],
                mean_length=result["mean_length"], elapsed=elapsed,
            )

        if step > 0 and step % save_freq == 0:
            agent.save(f"{save_dir}/step_{step}.pt")

    env.close()
    elapsed_total = time.time() - t_start
    total_frames = total_steps * env_cfg.frame_skip
    sps = (total_steps - start_step) / elapsed_total if elapsed_total > 0 else 0

    agent.q_network.eval()
    final = evaluate(agent, env_cfg, eval_episodes * 2)
    agent.q_network.train()

    agent.save(f"{save_dir}/final.pt")

    print()
    print("=" * 70)
    print(f"  TRAINING COMPLETE")
    print(f"  Steps: {total_steps:,}  |  Frames: {total_frames:,}")
    print(f"  Time: {elapsed_total:.0f}s ({sps:.0f} sps)")
    print(f"  Episodes: {episode_num}")
    print(f"  Best eval: {best_mean_reward:.1f}")
    print(f"  Final eval: {final['mean_reward']:.1f} +/- {final['std_reward']:.1f}")
    print("=" * 70)

    print(tracker.report(final["mean_reward"], best_mean_reward))

    return final


def print_targets():
    print()
    print("=" * 70)
    print("  RIVER RAID PERFORMANCE TARGETS")
    print("=" * 70)
    print(f"  {'Target':40s} {'Score':>12s}")
    print("-" * 70)
    print(f"  {'Theoretical Maximum (game score cap)':40s} {1_000_000:>12,}")
    print(f"  {'Rainbow @ 200M frames (SOTA)':40s} {20_675:>12,}")
    print(f"  {'Human Expert (median professional)':40s} {13_513:>12,}")
    print(f"  {'DQN Nature Paper':40s} {8_311:>12,}")
    print(f"  {'Rule-Based Heuristic':40s} {405:>12,}")
    print(f"  {'Random Agent':40s} {282:>12,}")
    print(f"  {'Current Best (Rainbow, 17.5K steps)':40s} {549:>12,}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="River Raid RL - Unified Training")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="Eval frequency")
    parser.add_argument("--save-freq", type=int, default=100_000, help="Save frequency")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Eval episodes")
    parser.add_argument("--run-name", type=str, default=None, help="Run name")
    parser.add_argument("--quick-test", action="store_true", help="Quick 10K test")
    parser.add_argument("--big-model", action="store_true", help="Use hidden_dim=512")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--no-clip", action="store_true", default=True, help="Disable reward clipping")
    parser.add_argument("--targets", action="store_true", help="Print performance targets")
    args = parser.parse_args()

    if args.targets:
        print_targets()
    else:
        train_riverraid(
            total_steps=args.steps,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            eval_episodes=args.eval_episodes,
            run_name=args.run_name,
            quick_test=args.quick_test,
            big_model=args.big_model,
            resume=args.resume,
            no_clip=args.no_clip,
        )
