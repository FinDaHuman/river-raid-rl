# Ablation Study – River Raid Rainbow DQN

This document describes how to run a full ablation study across all seven
Rainbow‑DQN components.  Each component can be toggled independently, and the
resulting agents can be compared using the scripts in `results/`.

## The Seven Components

| # | Component | Agent class | Config file | Toggle |
|---|-----------|-------------|-------------|--------|
| 0 | **Random** (baseline) | `RandomAgent` | – | random actions |
| 1 | **Vanilla DQN** | `DQNAgent` | `configs/dqn.yaml` | baseline |
| 2 | **Double DQN** | `DQNAgent` (DDQN is implicitly enabled in `update()`) | `configs/double_dqn.yaml` | – |
| 3 | **+ Dueling architecture** | `DuelingDQNAgent` | `configs/dueling_dqn.yaml` | use dueling stream |
| 4 | **+ Prioritized Replay** | `PERDQNAgent` | `configs/per_dqn.yaml` | use PER buffer |
| 5 | **+ C51 (Categorical)** | `C51DQNAgent` | `configs/c51.yaml` | use 51‑atom distribution |
| 6 | **Rainbow** (all above + N‑step) | `RainbowAgent` | `configs/rainbow.yaml` | full Rainbow |

## How to Run

```bash
# 1. Train each variant (this will take a while – run overnight)
python experiments/train.py -c configs/dqn.yaml         -o runs/dqn
python experiments/train.py -c configs/double_dqn.yaml  -o runs/double_dqn
python experiments/train.py -c configs/dueling_dqn.yaml -o runs/dueling_dqn
python experiments/train.py -c configs/per_dqn.yaml     -o runs/per_dqn
python experiments/train.py -c configs/c51.yaml         -o runs/c51
python experiments/train.py -c configs/rainbow.yaml     -o runs/rainbow

# 2. Generate the comparison table
python results/ablation_table.py

# 3. Plot training curves
python results/plot_training.py --paths runs/dqn runs/rainbow
```

## Expected Pattern

As you add components, you should observe:

- **Double DQN** → reduced over‑estimation bias (more stable learning curve)
- **Dueling** → better action‑value separation (faster convergence in early steps)
- **PER** → faster learning from rare/useful transitions
- **C51** → more stable returns (lower variance across episodes)
- **Rainbow** → the best overall performance, combining all benefits

## Results Template

After training all six agents, paste the table from `ablation_table.py` below:

```
(To be filled after training)
```
