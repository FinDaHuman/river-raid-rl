# 🎮 River Raid – Academic Reinforcement Learning Project

> *A clean, reproducible implementation of the Rainbow DQN algorithm, designed for low‑end hardware (laptops without dedicated GPUs) while showcasing a full RL research pipeline.*

---

## 📚 Project Overview

This repository contains a **self‑contained RL research suite** that trains an agent to play **Atari River Raid** using the **Rainbow DQN** algorithm (Hessel *et al.*, 2018).  The code is deliberately structured as a **monorepo** to:

1. **Demonstrate RL fundamentals** – each component of Rainbow (Double Q‑learning, Dueling architecture, Prioritized Experience Replay, N‑step returns, Categorical (C51) distributional RL, and the ε‑greedy schedule) is implemented as an independent, testable module.
2. **Enable academic reproducibility** – all hyper‑parameters are stored in human‑readable YAML files, the training loop logs metrics to JSON, and a LaTeX‑rich README explains the underlying mathematics.
3. **Run on low‑end hardware** – defaults target a typical laptop CPU (no GPU required).  Optional CUDA support is auto‑detected.

The repository follows the **standard RL research workflow**:

```
+-------------------+      +-------------------+      +-------------------+
|   Environment    | →    |   Agent (model)   | →    |   Replay Buffer   |
+-------------------+      +-------------------+      +-------------------+
        ^                          |                               |
        |                          v                               v
   Evaluation ←─────────────  Training Loop  ←─────────────  Sampling
```

---

## 🧪 Reinforcement Learning Foundations

### 1️⃣ Markov Decision Process (MDP)
The environment is modeled as an MDP $(\mathcal{S},\mathcal{A},P,R,\gamma)$ where:
* **$\mathcal{S}$** – set of stacked frames ($s_t \in \mathbb{R}^{4\times84\times84}$)
* **$\mathcal{A}=\{0\dots5\}$** – discrete actions (NOOP, FIRE, UP, RIGHT, LEFT, DOWN)
* **$P(s'\mid s,a)$** – transition dynamics defined by the Atari Learning Environment (ALE)
* **$R(s,a)$** – raw game score, **sign‑clipped** to $\{-1,0,+1\}$ (standard practice for Atari agents)
* **$\gamma$** – discount factor (default $0.99$)

### 2️⃣ Q‑learning Objective
The goal is to learn an action‑value function $Q_\theta(s,a)$ that satisfies the Bellman optimality equation:

$$
Q_\theta(s,a) = \mathbb{E}_{s'\sim P}\left[ R(s,a) + \gamma \max_{a'} Q_{\theta^-}(s',a') \right]
$$

where $\theta^-$ denotes the parameters of a **target network** that is periodically copied from the online network to stabilize learning.

### 3️⃣ Rainbow Enhancements
Rainbow synergistically combines six improvements:

| Component | Reference | Key Idea |
|---|---|---|
| **Double Q‑learning** | van Hasselt *et al.*, 2016 | Decouple action selection (online net) from evaluation (target net) to reduce over‑estimation bias. |
| **Dueling architecture** | Wang *et al.*, 2016 | Split \(Q\) into a state‑value stream \(V(s)\) and an advantage stream \(A(s,a)\): \(Q(s,a)=V(s)+A(s,a)-\frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a')\). |
| **Prioritized Experience Replay** | Schaul *et al.*, 2016 | Sample transitions with probability proportional to \(\big|\delta\big|^\alpha\) where \(\delta\) is the TD‑error. |
| **N‑step returns** | Peng *et al.*, 2018 | Bootstrap \(n\) steps ahead: \(R^{(n)}_t = \sum_{i=0}^{n-1}\gamma^i r_{t+i}+\gamma^n\max_{a'}Q(s_{t+n},a')\). |
| **Categorical (C51) Distributional RL** | Bellemare *et al.*, 2017 | Model the full distribution of returns \(Z(s,a)\) over \(\{z_i\}_{i=1}^K\) atoms. The loss is a *cross‑entropy* between projected target distribution and current distribution. |
| **NoisyNets (optional)** | Fortunato *et al.*, 2017 | Replace the ε‑greedy schedule with learned, state‑dependent noise in linear layers, encouraging exploration without hand‑tuned schedules. |

The **project’s ablation study** trains separate agents that enable/disable each component, allowing quantitative analysis of their impact on River Raid performance.

---

## 📦 Repository Layout (Monorepo)

```
riverraid-rl/
├─ pyproject.toml                # build & dependency metadata
├─ README.md                    # **this** document (LaTeX‑enabled)
├─ ABLATION.md                  # ablation‑study blueprint
├─ CONTRIBUTING.md              # developer guidelines
├─ CITATION.cff                 # academic citation metadata
├─ LICENSE / .gitignore / .gitattributes / .ruff.toml
├─ requirements.txt             # pip dependencies (optional; pyproject.toml preferred)
├─ configs/                     # YAML experiment specifications
│   ├─ rainbow.yaml             # full Rainbow DQN
│   └─ dqn.yaml                 # vanilla DQN (template for ablations)
├─ riverraid_rl/                # 🧠 core library (unchanged)
│   ├─ agents/   (Rainbow, DQN, baselines)
│   ├─ models/   (CNN, Dueling, Categorical)
│   ├─ memory/   (Replay, Prioritized, N‑Step)
│   ├─ env.py    (Atari wrappers)
│   ├─ config.py (dataclass configs)
│   └─ utils/    (evaluation, logger, progress tracker)
├─ experiments/                 # unified entry points
│   ├─ train.py                 # YAML‑driven trainer
│   ├─ evaluate.py              # checkpoint evaluation
│   ├─ train_cpu.py             # legacy CPU trainer
│   ├─ train_human_level.py     # human‑level baseline trainer
│   ├─ train_improved.py        # improved DQN variant
│   ├─ train_optimized.py       # optimised DQN variant
│   └─ train_riverraid.py       # original River Raid trainer
├─ scripts/                     # utility scripts (no training needed)
│   ├─ report_existing.py       # summarise all existing checkpoints
│   ├─ instantiate_agents.py    # verify all agents can be built
│   └─ render_gameplay.py       # render agent gameplay video
├─ results/                     # analysis & visualisation
│   ├── plot_training.py        # training curves from logged metrics
│   └── ablation_table.py       # comparison table from progress.json
├─ notebooks/                   # interactive tutorials
│   └─ quickstart.ipynb         # walk‑through notebook
├─ tests/                       # pytest suite
│   ├─ test_agents_instantiate.py
│   ├─ test_instantiate_exported.py
│   ├─ test_config.py
│   └─ … (original test files)
├─ docs/                        # MkDocs documentation source
│   ├─ index.md                 # MkDocs homepage
│   ├─ PLAN_HUMAN_LEVEL.md      # human‑level training plan
│   └─ RIVERRAID_RL_REPORT.md   # project report
├─ mkdocs.yml                   # documentation site config
├─ .github/workflows/tests.yml  # CI pipeline
└─ media/                       # demo video
```

---

## 🚀 Getting Started

### Prerequisites
```bash
# Install Python ≥3.10
python -m pip install -r requirements.txt   # optional – we also ship a pyproject.toml
```
The required packages are listed in ``pyproject.toml`` (torch, gymnasium[atari], ale-py, numpy, pyyaml).

### Quick Run (CPU) → Train a Rainbow Agent
```bash
# 1. Create a virtualenv (recommended)
python -m venv .venv && source .venv/bin/activate   # on Windows use .venv\Scripts\activate

# 2. Install dependencies
pip install -e .   # editable install reads pyproject.toml

# 3. Train using the supplied config (runs on CPU if no GPU detected)
python experiments/train.py -c configs/rainbow.yaml -o runs/rainbow_demo
```
The script will:
- Build the ALE River Raid environment with standard preprocessing (frame‑stack, greyscale, resize).
- Instantiate a **RainbowAgent** with the hyper‑parameters from ``configs/rainbow.yaml``.
- Log training metrics to ``runs/rainbow_demo/logs/metrics.json``.
- Save checkpoints (including the best‑performing model) in ``runs/rainbow_demo/``.
- Print a concise progress table after each evaluation phase.

### Evaluation of a Saved Checkpoint
```bash
python experiments/evaluate.py \
    --checkpoint runs/rainbow_demo/best_500000.pt \
    --episodes 20
```
The script loads the checkpoint, runs the agent **greedy** for the requested number of episodes (no reward clipping) and reports mean/std reward, episode length, and performance relative to the academic benchmarks defined in ``ProgressTracker``.

---

## 📊 Expected Benchmarks (low‑end hardware)
| Metric | Target | Current best (CPU) |
|---|---|---|
| **Theoretical max** | 1 000 000 pts | – |
| **Human expert** | 13 513 pts | ~588 (≈4.35 % of human) |
| **Rainbow SOTA** (200 M frames) | 20 675 pts | – |
| **DQN Nature (2015)** | 8 311 pts | – |

The **ProgressTracker** (see ``riverraid_rl/utils/progress.py``) automatically computes percentages against these targets after each evaluation.

---

## 📊 Analysis Without Training

These scripts run **immediately** – no environment, no GPU, no training needed:

| Script | What it does | Run it |
|--------|-------------|--------|
| `scripts/report_existing.py` | Scans all `checkpoints/*/progress.json` and prints a compact performance table | `python scripts/report_existing.py` |
| `scripts/instantiate_agents.py` | Imports every agent, instantiates it, calls `act()` on dummy data | `python scripts/instantiate_agents.py` |
| `results/plot_training.py` | Reads logged metrics and generates publication‑quality PNGs | `python results/plot_training.py` |
| `results/ablation_table.py` | Reads `progress.json` files and prints a Markdown comparison table | `python results/ablation_table.py` |

```bash
# Example – see what the current checkpoints have achieved:
python scripts/report_existing.py
```

## 🧪 Ablation Study Blueprint

A detailed ablation guide is in **[ABLATION.md](ABLATION.md)**.  In short:

1. Each Rainbow component can be toggled independently via its own YAML config.
2. After training, `results/ablation_table.py` produces a comparison table.
3. `results/plot_training.py` draws learning curves overlaid for all runs.

## 📘 Quickstart Notebook

Open `notebooks/quickstart.ipynb` for an interactive walk‑through:

```bash
jupyter notebook notebooks/quickstart.ipynb
```

It covers:
- Loading a config file
- Instantiating an agent
- Running a single evaluation episode
- Plotting the result

---

## 📜 Academic References
- **Rainbow** – Hessel, M., et al. *Rainbow: Combining Improvements in Deep Reinforcement Learning*. AAAI 2018.
- **C51** – Bellemare, M. G., et al. *A Distributional Perspective on Reinforcement Learning*. ICML 2017.
- **Dueling DQN** – Wang, Z., et al. *Dueling Network Architectures for Deep Reinforcement Learning*. ICML 2016.
- **Prioritized Replay** – Schaul, T., et al. *Prioritized Experience Replay*. ICLR 2016.
- **Noisy Nets** – Fortunato, M., et al. *Noisy Networks for Exploration*. ICLR 2018.
- **Atari Learning Environment** – Bellemare, M. G., et al. *The Arcade Learning Environment*. JAI 2013.

---

## 🧑‍💻 Contribution Guidelines
1. **Branch‑first workflow** – create a feature branch for each experimental variant.
2. **Tests** – new functionality must be accompanied by unit tests in ``tests/``.
3. **Documentation** – update the relevant section of this README (including LaTeX formulas) when adding algorithmic components.
4. **Reproducibility** – commit the exact YAML config used for any published result.

---

## 📚 License
This project is licensed under the MIT License (see ``LICENSE``).

---

*Happy hacking, and may your agents achieve ever‑higher scores!*
