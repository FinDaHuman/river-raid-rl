import argparse
import warnings
import time

import numpy as np

warnings.filterwarnings("ignore")

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from riverraid_rl.agents.dqn import DQNAgent
from riverraid_rl.agents.rainbow import RainbowAgent
from riverraid_rl.agents.icm_rainbow import ICMRainbowAgent
from riverraid_rl.agents.hierarchical import HierarchicalRiverRaidAgent
from riverraid_rl.agents.rule_based import RuleBasedAgent
from riverraid_rl.agents.random_agent import RandomAgent
from riverraid_rl.config import Config
from riverraid_rl.env import make_riverraid_env, get_minimal_action_set
from riverraid_rl.utils.evaluation import evaluate
from riverraid_rl.train import train
from riverraid_rl.pbt import PBTTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval", "baselines", "pbt", "profile", "benchmark"])
    parser.add_argument("--agent", type=str, default="rainbow",
                        choices=["dqn", "rainbow", "icm_rainbow", "hierarchical", "random", "rule"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=10000000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--pbt_pop", type=int, default=4)
    parser.add_argument("--pbt_gen", type=int, default=3)
    parser.add_argument("--steps_per_gen", type=int, default=20000)
    args = parser.parse_args()

    config = Config()
    config.training.total_timesteps = args.timesteps
    config.training.device = args.device
    config.training.run_name = f"{args.agent}-{args.timesteps // 1000000}m-{int(time.time())}"
    num_actions = 6

    if args.mode == "benchmark":
        print("=== Full Agent Benchmark on Riverraid ===\n")
        agent_configs = [
            ("Random", RandomAgent(num_actions), {}),
            ("Rule-Based", RuleBasedAgent(), {}),
            ("DQN", DQNAgent(config.env, config.dqn, num_actions, args.device),
             {"checkpoint": f"checkpoints/dqn-{args.timesteps // 1000000}m/best.pt"}),
            ("Rainbow", RainbowAgent(config.env, config.rainbow, num_actions, args.device),
             {"checkpoint": f"checkpoints/rainbow-{args.timesteps // 1000000}m/best.pt"}),
            ("ICM-Rainbow", ICMRainbowAgent(config.env, config.rainbow, num_actions, args.device, args.eta),
             {"checkpoint": f"checkpoints/icm_rainbow-{args.timesteps // 1000000}m/best.pt"}),
            ("Hierarchical", HierarchicalRiverRaidAgent(config.env, config.rainbow, num_actions, args.device, args.eta),
             {"checkpoint": f"checkpoints/hierarchical-{args.timesteps // 1000000}m/best.pt"}),
        ]

        results = []
        for name, agent, kwargs in agent_configs:
            if kwargs.get("checkpoint"):
                try:
                    agent.load(kwargs["checkpoint"])
                    print(f"Loaded checkpoint for {name}")
                except:
                    print(f"No checkpoint for {name}, using untrained agent")

            result = evaluate(agent, config.env, args.episodes)
            results.append((name, result))
            print(f"{name:20s} | Reward: {result['mean_reward']:8.2f} +/- {result['std_reward']:6.2f} | "
                  f"Length: {result['mean_length']:6.1f} | Episodes: {result['num_episodes']}")

        print(f"\n{'='*60}")
        print(f"{'Agent':20s} {'Mean Reward':>12s} {'Std':>8s} {'Mean Length':>12s}")
        print(f"{'='*60}")
        for name, result in sorted(results, key=lambda x: x[1]["mean_reward"], reverse=True):
            print(f"{name:20s} {result['mean_reward']:8.2f}  +/-{result['std_reward']:6.2f}  {result['mean_length']:8.1f}")
        return

    if args.mode == "baselines":
        print("=== Running Baseline Agents on Riverraid ===")
        for name, agent_cls in [("Random", RandomAgent), ("Rule-Based", RuleBasedAgent)]:
            if name == "Random":
                agent = agent_cls(num_actions)
            else:
                agent = agent_cls()
            print(f"\n--- {name} Agent ---")
            result = evaluate(agent, config.env, args.episodes)
            print(f"  Mean Reward: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}")
            print(f"  Mean Length: {result['mean_length']:.1f} steps")
        return

    if args.mode == "pbt":
        print("=== Running Population-Based Training ===")
        trainer = PBTTrainer(population_size=args.pbt_pop, num_generations=args.pbt_gen)
        best_agent = trainer.run(config, num_steps_per_gen=args.steps_per_gen)
        best_agent.save(f"checkpoints/pbt-best.pt")
        result = evaluate(best_agent, config.env, args.episodes)
        print(f"PBT Best Agent: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}")
        return

    if args.mode == "profile":
        print("=== Profiling Agent ===")
        agent = RainbowAgent(config.env, config.rainbow, num_actions, args.device)
        env = make_riverraid_env(config.env)
        state, info = env.reset()
        start = time.time()
        num_actions_taken = 0
        for _ in range(1000):
            action = agent.act(np.array(state), training=True)
            state, reward, terminated, truncated, info = env.step(action)
            num_actions_taken += 1
            if terminated or truncated:
                state, info = env.reset()
        elapsed = time.time() - start
        print(f"Actions taken: {num_actions_taken}")
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Actions/second: {num_actions_taken / elapsed:.1f}")
        env.close()
        return

    if args.agent == "dqn":
        agent = DQNAgent(config.env, config.dqn, num_actions, args.device)
    elif args.agent == "rainbow":
        agent = RainbowAgent(config.env, config.rainbow, num_actions, args.device)
    elif args.agent == "icm_rainbow":
        agent = ICMRainbowAgent(config.env, config.rainbow, num_actions, args.device, eta=args.eta)
    elif args.agent == "hierarchical":
        agent = HierarchicalRiverRaidAgent(config.env, config.rainbow, num_actions, args.device, eta=args.eta)
    elif args.agent == "random":
        agent = RandomAgent(num_actions)
    elif args.agent == "rule":
        agent = RuleBasedAgent()
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    if args.checkpoint:
        agent.load(args.checkpoint)

    if args.mode == "train":
        train(agent, config)
    elif args.mode == "eval":
        result = evaluate(agent, config.env, args.episodes)
        print("Evaluation Results:")
        for key, value in result.items():
            print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    main()
