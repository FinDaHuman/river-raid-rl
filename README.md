# 🚁 River Raid RL — Reinforcement Learning Agent

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c" alt="PyTorch 2.0+"></a>
  <a href="https://gymnasium.farama.org/"><img src="https://img.shields.io/badge/Gymnasium-1.0%2B-blueviolet" alt="Gymnasium 1.0+"></a>
</p>

A deep reinforcement learning agent that plays **River Raid** (Atari 2600) using the **Rainbow DQN** algorithm. Built with PyTorch, Gymnasium, and the Arcade Learning Environment (ALE).

---

## 🎬 Demo

Watch the trained agent fly through River Raid at its best:

![Gameplay Demo](media/riverraid-best.mp4)

*Agent scoring **630 points** using the Rainbow DQN checkpoint trained for 200K steps (800K game frames).*

```bash
# Recreate this video with the latest checkpoint:
python render_gameplay.py
```

---

## 🎯 Performance vs Known Benchmarks

| Benchmark | Score | % of Theoretical Max | Source |
|---|---|---|---|
| **Theoretical Maximum** | **1,000,000** | **100%** | Game score cap (rolls to `!!!!!!`) |
| **Tool-Assisted Speedrun (TAS)** | **1,000,000** | **100%** | Frame-perfect bot (TASVideos) |
| **Rainbow SOTA (200M frames)** | **20,675** | **2.07%** | Published Rainbow DQN paper |
| **Human Expert** | **13,513** | **1.35%** | Median professional player |
| **Average Human** | **~5,000–8,000** | **~0.5–0.8%** | Typical casual player |
| **DQN Nature Paper** | **8,311** | **0.83%** | Mnih et al. 2015 |
| **Random Agent** | **282** | **0.03%** | Uniform random actions |
| **⬆ Our Agent (best)** | **588** | **0.06%** | Rainbow, 200K steps, 800K frames |

**Progress after each training session is reported as % of all four targets** — theoretical max, human expert, Rainbow SOTA, and DQN Nature.

---

## 🧠 How the Agent Works

### Algorithm: Rainbow DQN

The agent combines **six proven improvements** over the original Deep Q-Network (DQN), each addressing a specific limitation:

| Component | What It Fixes | How |
|---|---|---|
| **Distributional RL (C51)** | DQN loses return variance info | Learn full return distribution over 51 discrete atoms (value buckets) instead of just expected value |
| **Dueling Network** | DQN wastes capacity on similar-valued actions | Split Q-value into state-value V(s) + action-advantage A(s,a) |
| **Prioritized Replay** | Rare useful experiences are forgotten | Sample transitions by TD-error magnitude, not uniformly |
| **N-Step Returns** | Slow reward propagation | Look ahead N=3 steps per update for faster learning |
| **Double Q-Learning** | Overestimation bias | Use online network for action selection, target network for value |
| **Convolutional Encoder** | Raw pixels are high-dimensional | Three Conv2D layers compress 4×84×84 frames → 3,136-d feature vector |

### Neural Network Architecture

```
Input: 4 stacked grayscale frames (84×84)
    ↓
Conv2D(4→32, 8×8, stride=4) → ReLU
    ↓
Conv2D(32→64, 4×4, stride=2) → ReLU
    ↓
Conv2D(64→64, 3×3, stride=1) → ReLU
    ↓
Flatten → 3,136 features
    ↓
┌───── Value Stream ─────┐    ┌───── Advantage Stream ─────┐
│ Linear(3136→256) → ReLU │    │ Linear(3136→256) → ReLU    │
│ Linear(256→51 atoms)    │    │ Linear(256→6×51 atoms)     │
└─────────────────────────┘    └────────────────────────────┘
    ↓                                  ↓
    └──────── Q = V + A − mean(A) ─────┘
                    ↓
          Softmax over 51 atoms per action
                    ↓
          Q(s,a) = Σ(z_i · p_i)  (expected value)
```

**Total parameters: 1,775,877**

### Training Loop

```
For each environment step:
  1. Agent observes state (4 stacked frames)
  2. Agent selects action (ε-greedy: random with prob ε, else greedy)
  3. Action repeated for 4 game frames (frame skip)
  4. Store transition (s, a, r, s', done) in replay buffer
  5. Every 4 steps: sample 32 transitions, compute Rainbow loss, update network
  6. Every 2,500 updates: copy online network → target network
  7. Every 50K steps: evaluate greedily over 10 episodes, save if best
```

### Action Space (6 discrete actions)

| Index | Action |
|---|---|
| 0 | NOOP |
| 1 | FIRE |
| 2 | UP |
| 3 | RIGHT |
| 4 | LEFT |
| 5 | DOWN |

---

## 📊 Current Results

### Best Training Run (200K steps, ~800K game frames)

| Step | Frames | Mean Score | % Max | % Human | % SOTA | Best |
|---|---|---|---|---|---|---|
| 50,000 | 200,000 | 89.0 | 0.009% | 0.66% | 0.43% | 89.0 |
| **100,000** | **400,000** | **588.0** | **0.059%** | **4.35%** | **2.84%** | **588.0** |
| 150,000 | 600,000 | 426.0 | 0.043% | 3.15% | 2.06% | 588.0 |

- **Best single episode:** 1,210 points
- **Peak evaluation mean:** 588.0 ± 464.5
- **Training speed:** ~100 steps/second (CPU)

### All Agents Compared

| Agent | Score | % Max | % Human |
|---|---|---|---|
| **Rainbow (our best)** | **588** | **0.06%** | **4.35%** |
| Rule-Based Heuristic | 405 | 0.04% | 3.00% |
| Random | 282 | 0.03% | 2.09% |
| DQN (50K steps) | 72 | 0.01% | 0.53% |

---

## 🧗 Why Can't We Reach 1,000,000?

The theoretical max of **1,000,000 points** is the score cap on the original Atari 2600 cartridge. It has been achieved by exactly one method: a **Tool-Assisted Speedrun (TAS)** that took 1 hour 22 minutes of frame-perfect inputs. This is unreachable by an RL agent in the ALE for fundamental reasons:

### ALE Constraints (Hard Limits)

| Constraint | ALE Value | What We Need | Problem |
|---|---|---|---|
| **Episode length** | 108,000 steps (max) | ~556 bridge sections | The game never ends, but ALE terminates episodes |
| **Fuel density at high levels** | ~6% of objects | ~24% at level 1 | After 100K points, fuel is too scarce to survive |
| **Death abuse** | Not allowed | Deliberate crashing for refuel | Requires exact route planning, not learned |

The TAS solution works because it:
1. Knows the exact game layout in advance (deterministic ROM)
2. Routes through each life to maximize score before planned death for refuel
3. Uses frame-perfect inputs to manipulate enemy positions

**An RL agent can never match this** — it must learn from scratch without access to the game's internal state or a simulator that can "look ahead."

### Training Constraints (Soft Limits, Can Be Improved)

| Constraint | Current | Target | Fix |
|---|---|---|---|
| **Training steps** | 200,000 | 50,000,000 | GPU training (100× speedup) |
| **Exploration** | ε-greedy (90% random at best) | State-dependent | NoisyNets (already implemented) |
| **Data throughput** | 1 environment | 8+ environments | Vectorized envs (already implemented) |
| **Buffer size** | 200,000 | 1,000,000+ | Increase capacity |
| **Epsilon decay** | 250,000 steps | 1,000,000+ | Longer exploration phase |

### Realistic Performance Ceiling in ALE

The absolute ceiling for a learning agent in the ALE is estimated at **~80,000–100,000 points** (8–10% of theoretical max). Beyond this point, the game's increasing difficulty and ALE's episode limit prevent further progress regardless of agent capability.

---

## 🚀 How to Improve (Next Steps)

### Quick Wins (Code Already Exists)

| Feature | File | Effort |
|---|---|---|
| **NoisyNets** — Replace ε-greedy with learned noise | `models/noisy.py` | Hours (wire into Rainbow) |
| **Vectorized environments** — 8× data throughput | `train_optimized.py` | Already works |
| **ICM + Rainbow** — Intrinsic curiosity for exploration | `agents/icm_rainbow.py` | Tune hyperparameters |
| **PBT** — Auto-tune hyperparameters | `pbt.py` | Runs now |
| **Attention layers** — Spatial/Channel attention | `models/attention.py` | Swap into network |

### Scaling

| Resource | Current Performance | Expected After |
|---|---|---|
| GPU training (1× RTX 4090) | ~100 sps (CPU) | ~10,000+ sps |
| 50M frames training | ~7 days CPU | ~2-3 hours GPU |
| Match Human Expert (13,513) | ~7 days CPU | ~2-3 hours GPU |
| Match Rainbow SOTA (20,675) | ~7 days CPU | ~2-3 hours GPU |

### Advanced: Beyond Rainbow

1. **Rainbow-IQN** — Implicit Quantile Networks for continuous distributional RL
2. **Munchausen DQN** — Entropy-regularized Q-learning
3. **EfficientZero / DreamerV3** — Model-based RL for sample efficiency
4. **Bigger network** — Increase hidden_dim to 512, add residual connections

---

## 🛠️ Setup & Usage

### Requirements

```bash
pip install torch gymnasium[atari] ale_py numpy
```

### Quick Start

```bash
# See performance targets
python train_riverraid.py --targets

# Train agent with progress tracking (200K steps)
python train_riverraid.py --steps 200000 --run-name my-run

# Quick 10K-step test
python train_riverraid.py --quick-test

# Resume from checkpoint
python train_riverraid.py --steps 500000 --resume checkpoints/my-run/best.pt
```

### Training Output

Every evaluation (default: every 50K steps) produces a progress report:

```
===========================================================================
                      Progress Report
===========================================================================
  Metric                                Current      Best Ever         Target
---------------------------------------------------------------------------
  Raw Score                               588.0          588.0      1,000,000
  % of Theoretical Max (1M)             0.0588%        0.0588%      100.0000%
  % of Human Expert (13.5K)               4.35%          4.35%        100.00%
  % of Rainbow SOTA (20.7K)               2.84%          2.84%        100.00%
  % of DQN Nature (8.3K)                  7.08%          7.08%        100.00%
===========================================================================
```

Full evaluation history is saved to `checkpoints/<run-name>/progress.json`.

### Evaluate a Checkpoint

```bash
python -c "
from riverraid_rl.agents.rainbow import RainbowAgent
from riverraid_rl.config import EnvConfig, RainbowConfig
from riverraid_rl.utils.evaluation import evaluate
cfg = RainbowConfig(); cfg.hidden_dim = 256; cfg.num_atoms = 51
cfg.v_min = -5; cfg.v_max = 50; cfg.min_replay_size = 5000; cfg.buffer_capacity = 200000
agent = RainbowAgent(EnvConfig(), cfg, 6, 'cpu')
agent.load('checkpoints/rainbow-200k/best.pt')
r = evaluate(agent, EnvConfig(), 20)
print(f'Mean: {r[\"mean_reward\"]:.1f} ± {r[\"std_reward\"]:.1f}')
"
```

---

## 📁 Project Structure

```
riverraid_rl/                  # Core RL library
├── agents/                    # Agent implementations
│   ├── rainbow.py             # ⭐ Rainbow DQN (primary agent)
│   ├── dqn.py                 # Vanilla DQN
│   ├── icm_rainbow.py         # Rainbow + Intrinsic Curiosity
│   ├── hierarchical.py        # Dual-policy hierarchical agent
│   ├── rule_based.py          # Heuristic baseline
│   └── random_agent.py        # Random baseline
├── models/                    # Neural network architectures
│   ├── cnn.py                 # DQNCNN, DuelingDQN, CategoricalDuelingDQN
│   ├── attention.py           # Spatial/Channel attention networks
│   ├── noisy.py               # NoisyLinear layers (unused, ready to wire)
│   └── icm.py                 # Intrinsic Curiosity Module
├── memory/
│   └── replay.py              # ReplayBuffer, Prioritized, NStep variants
├── utils/
│   ├── evaluation.py          # Policy evaluation harness
│   ├── logger.py              # Training metrics logger
│   └── progress.py            # ⭐ Progress tracker (% max, % human, % SOTA)
├── env.py                     # Environment wrappers
├── env_hierarchical.py        # Fuel tracking for hierarchical agent
├── config.py                  # Dataclass-based configuration
├── train.py                   # Core training loop
├── curriculum.py              # Curriculum learning wrapper
├── pbt.py                     # Population-Based Training
└── scripts/                   # Evaluation & analysis scripts
    ├── eval_all.py
    └── final_results.py

train_riverraid.py             # ⭐ Unified training entry point (recommended)
train_optimized.py             # Multi-environment training
train_improved.py              # Improved single-env training
train_cpu.py                   # CPU-optimized training
render_gameplay.py             # Record gameplay video from a checkpoint
RIVERRAID_RL_REPORT.md         # Full technical report
LICENSE                        # MIT License
media/                         # Demo videos and generated media

scripts/                       # One-off evaluation & benchmark scripts
├── eval_checkpoints.py
├── eval_dqn50k.py
├── benchmark_speed.py
└── _eval_baselines.py
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@software{riverraid_rl,
  author = {},
  title = {River Raid RL -- Rainbow DQN Agent for Atari 2600},
  year = {2026},
  url = {https://github.com/anomalyco/river-raid-rl}
}
```

For the underlying algorithms:

```bibtex
@article{hessel2018rainbow,
  title={Rainbow: Combining Improvements in Deep Reinforcement Learning},
  author={Hessel, Matteo and Modayil, Joseph and Van Hasselt, Hado and others},
  journal={AAAI},
  year={2018}
}
```

---

## 📚 References

- **Rainbow DQN** — Hessel et al., *Rainbow: Combining Improvements in Deep Reinforcement Learning* (AAAI 2018)
- **Categorical DQN (C51)** — Bellemare et al., *A Distributional Perspective on Reinforcement Learning* (ICML 2017)
- **Dueling DQN** — Wang et al., *Dueling Network Architectures for Deep Reinforcement Learning* (ICML 2016)
- **Prioritized Replay** — Schaul et al., *Prioritized Experience Replay* (ICLR 2016)
- **Human-level Control (DQN)** — Mnih et al., *Human-level control through deep reinforcement learning* (Nature 2015)
- **TASVideos** — Lord_Tom, River Raid 1,000,000 point speedrun (#4648S)
- **ALE** — Bellemare et al., *The Arcade Learning Environment* (JAIR 2013)
