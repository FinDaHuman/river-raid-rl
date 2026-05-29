import copy
import random

import numpy as np
import torch

from riverraid_rl.agents.rainbow import RainbowAgent
from riverraid_rl.config import Config, RainbowConfig
from riverraid_rl.env import make_riverraid_env
from riverraid_rl.utils.evaluation import evaluate


class PBTTrainer:
    def __init__(
        self,
        population_size: int = 8,
        num_generations: int = 5,
        exploit_quantile: float = 0.2,
        explore_prob: float = 0.2,
    ):
        self.population_size = population_size
        self.num_generations = num_generations
        self.exploit_quantile = exploit_quantile
        self.explore_prob = explore_prob

    def _sample_hyperparams(self) -> dict:
        return {
            "learning_rate": float(np.random.uniform(1e-5, 1e-3)),
            "batch_size": int(np.random.choice([32, 64, 128])),
            "target_update_freq": int(np.random.choice([2000, 5000, 10000])),
            "gamma": float(np.random.uniform(0.95, 0.999)),
            "hidden_dim": int(np.random.choice([256, 512, 1024])),
            "num_atoms": int(np.random.choice([51, 101, 201])),
        }

    def _perturb(self, params: dict) -> dict:
        new_params = copy.deepcopy(params)
        for key in params:
            if random.random() < self.explore_prob:
                if isinstance(params[key], float):
                    new_params[key] *= np.random.uniform(0.8, 1.2)
                elif isinstance(params[key], int):
                    new_params[key] = int(params[key] * np.random.uniform(0.8, 1.2))
        return new_params

    def run(self, config: Config, num_steps_per_gen: int = 50000) -> RainbowAgent:
        population = []
        scores = []

        for i in range(self.population_size):
            hp = self._sample_hyperparams()
            cfg = Config()
            cfg.rainbow.learning_rate = hp["learning_rate"]
            cfg.rainbow.batch_size = hp["batch_size"]
            cfg.rainbow.target_update_freq = hp["target_update_freq"]
            cfg.rainbow.gamma = hp["gamma"]
            cfg.rainbow.hidden_dim = hp["hidden_dim"]
            cfg.rainbow.num_atoms = hp["num_atoms"]
            cfg.training.run_name = f"pbt-member-{i}"

            agent = RainbowAgent(cfg.env, cfg.rainbow, 6, config.training.device)
            population.append((agent, hp))
            scores.append(0.0)

        for gen in range(self.num_generations):
            print(f"\n=== PBT Generation {gen + 1} ===")

            for i, (agent, hp) in enumerate(population):
                print(f"  Training member {i}...")
                self._train_agent(agent, config, num_steps_per_gen)
                result = evaluate(agent, config.env, num_episodes=5)
                scores[i] = result["mean_reward"]
                print(f"    Score: {scores[i]:.2f}")

            threshold = np.quantile(scores, self.exploit_quantile)
            for i in range(self.population_size):
                if scores[i] < threshold:
                    top_idx = np.argmax(scores)
                    agent, hp = population[i]
                    top_agent, top_hp = population[top_idx]
                    agent.q_network.load_state_dict(top_agent.q_network.state_dict())
                    agent.target_network.load_state_dict(top_agent.target_network.state_dict())
                    hp = self._perturb(top_hp)
                    cfg = Config()
                    for key, value in hp.items():
                        setattr(cfg.rainbow, key, value)
                    agent.config = cfg.rainbow
                    population[i] = (agent, hp)
                    print(f"    Replaced member {i} with best (score {scores[top_idx]:.2f})")

        best_idx = np.argmax(scores)
        return population[best_idx][0]

    def _train_agent(self, agent, config, num_steps):
        env = make_riverraid_env(config.env)
        state, info = env.reset()
        for step in range(num_steps):
            action = agent.act(np.array(state), training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, terminated or truncated)
            state = next_state
            if step >= agent.config.min_replay_size and step % agent.config.train_freq == 0:
                agent.update()
            if terminated or truncated:
                state, info = env.reset()
        env.close()
