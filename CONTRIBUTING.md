# Contributing to River Raid RL

Thank you for your interest! This project aims to be a reproducible, academically‑rigorous reinforcement‑learning research suite. Contributions that improve clarity, expand the ablation study, or strengthen reproducibility are especially welcome.

## Workflow

1. **Branch‑first:**  
   `git checkout -b feat/your-feature`  
   Never commit directly to `main`.

2. **Code style:**  
   We use [ruff](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`).  
   Run `ruff check .` before committing.

3. **Tests:**  
   All new functionality must include `pytest` tests.  
   Run the full suite: `python -m pytest tests/ -v`

4. **Configs:**  
   Every experiment must have a YAML file in `configs/`.  
   Commit the exact YAML used for any published result.

5. **Documentation:**  
   Update `ABLATION.md` if you add or modify an agent variant.  
   Add LaTeX equations to `README.md` when introducing new algorithms.

## Adding a New Agent Variant

1. Create the agent class in `src/riverraid_rl/agents/` (or directly in `riverraid_rl/agents/` for the original codebase).
2. Add a corresponding YAML config in `configs/`.
3. Register the agent in `experiments/train.py`’s `build_agent()` factory.
4. Add a unit test in `tests/`.
5. Run `python results/ablation_table.py` after training to compare results.

## Reporting Issues

Open a GitHub issue with:
- A minimal reproduction script
- The exact YAML config used
- The full error message (if applicable)
