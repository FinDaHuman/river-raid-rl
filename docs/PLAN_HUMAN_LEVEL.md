# Plan: Reaching Human-Level Play in River Raid

**Target:** 13,513 points (human expert) → 20,675 (Rainbow SOTA)
**Current:** 588 eval mean (best checkpoint, 200K steps)
**Gap:** 23× improvement needed

**Implementation status:** Code changes complete ✓ — LR cosine decay, v_max=50, buffer=100K, min_replay=20K, target_update_freq=8K. All 7 tests passing.

**Real (not estimated) throughput on RTX 3050 Ti Laptop:**
| Envs | Steps/s | Frames/s | GPU mem |
|---|---|---|---|
| 1 | 140 | 558 | 111 MB |
| 4 | 78 | 1,241 | 112 MB |
| 8 | 46 | 1,460 | 110 MB |

**Bottleneck: ALE game emulation (CPU-bound), not GPU.** Neural net only 27ms/update. 110 MB of 4 GB GPU used.

**12-hour estimate:** 63M frames, expected score ~7K-10K (may fall short of 13,513 human expert — SOTA needed 200M frames for 20,675 points).

---

## Phase 0: Critical Hyperparameter Fixes (Before Any Training)

The current `train_optimized.py` and `BetterThanHumanRainbowConfig` have bugs that prevent reaching high scores regardless of training duration.

### 0.1 — v_max is too low in training script

The `BetterThanHumanRainbowConfig` class default `v_max=100` is correct, but `train_optimized.py` **overrides it to 10.0** — this is the bug.

| Location | Current (bug) | Fixed to | Why |
|---|---|---|---|
| `train_optimized.py` config override | v_max=10.0 | v_max=50.0 | With clipped rewards {-1,0,1} and gamma=0.99^n, max return is ~100. v_max=50 covers mid-training range comfortably. 51 atoms spaced 1.2 apart are fine-grained enough. |

**Why v_max=10 was wrong:** The C51 atom distribution can only represent Q-values in [v_min, v_max]. With clipped rewards, a good state's Q-value easily exceeds 10, collapsing all probability to the top atom. The network loses all discriminative ability between states.

### 0.2 — Buffer capacity too small

| Current (bug) | Fixed to | Why |
|---|---|---|
| 50,000 | 100,000 (RAM-limited) | Published Rainbow uses 1M. With 4 envs at 1,686 fps, 100K fills in ~1 min. System RAM constraint: 200K+ would need >10 GB contiguous. 100K is a practical 2× improvement that fits. |

### 0.3 — Min replay size too low

| Current (bug) | Fixed to | Why |
|---|---|---|
| 5,000 | 20,000 | Published Rainbow uses 80,000. 20K ensures diverse initial data without wasting too many steps. |

### 0.4 — Target update frequency too aggressive

| Current (bug) | Fixed to | Why |
|---|---|---|
| 2,500 updates | 8,000 updates | Frequent target updates destabilize learning. Published Rainbow uses 32,000 updates. 8K is a middle ground for shorter runs. |

### 0.5 — Learning rate cosine decay

Add cosine annealing LR schedule to the agent. Start at `learning_rate` and decay to 0 across total steps. This stabilizes late training and prevents divergence.

**Implementation:** Add `lr_scheduler` to `BetterThanHumanRainbowAgent.__init__`:
```python
self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
    self.optimizer, T_max=total_steps
)
```
Call `self.scheduler.step()` after each optimizer step.

### 0.6 — Gamma mismatch with n-step

| Parameter | Current | Recommended | Why |
|---|---|---|---|
| `gamma` | 0.997 | 0.99 | γ=0.997 matches 5-step bootstrap (γ^n = 0.985). But with long episodes (10K+ steps), effective horizon is ~3,300 steps. This is fine but the published Rainbow uses γ=0.99 with 3-step. Either works — keep 0.997 for 5-step consistency. |

### 0.7 — Reward clipping

Current: `clip_rewards=True` in `train_optimized.py` (sign clipping: rewards become {-1, 0, 1}). Published Rainbow uses sign clipping. **Keep this** — it's standard and proven.

### 0.8 — NoisyNets vs epsilon

The `BetterThanHumanRainbowAgent` uses NoisyNets (replaces epsilon-greedy). **This is correct.** NoisyNets provide state-dependent exploration and don't need a decay schedule. However, the agent needs a warm-start phase where it acts randomly to fill the buffer before NoisyNet weights converge.

---

## Phase 1: Baseline Validation (1-2 hour run)

Prove the tuned hyperparameters don't break training.

**Training config:**
| Parameter | Value |
|---|---|
| Steps | 500,000 |
| Envs | 4 |
| v_min / v_max | -10 / 50 |
| Buffer capacity | 500,000 |
| Min replay size | 20,000 |
| Target update freq | 8,000 |
| Learning rate | 0.0001 (cosine decay) |
| Frame skip | 4 |
| Total game frames | 8,000,000 |
| Expected time on GPU | ~80 min |
| Expected score | ~2,000-4,000 (estimate) |

**Success criteria:**
- Mean eval score > 1,000 (2× current best)
- No NaN losses, stable Q-values
- Throughput consistent at ~100 sps

**Run command:**
```bash
python train_optimized.py --steps 500000 --envs 4 --eval-freq 50000
```

---

## Phase 2: Medium Run — Probe Human-Level (8-10 hour run)

If Phase 1 passes, scale to the first serious attempt at human level.

**Training config:**
| Parameter | Value |
|---|---|
| Steps | 3,000,000 |
| Envs | 4 |
| v_min / v_max | -10 / 200 |
| Buffer capacity | 500,000 |
| Target update freq | 2,500 |
| Total game frames | 48,000,000 |
| Expected time on GPU | ~8 hours |
| Expected score | ~5,000-10,000 (estimate) |

**Success criteria:**
- Mean eval score > 5,000 (37% of human expert)
- Learning curve still trending upward at plateau
- Max single-episode score > 10,000

**Checkpoints:**
- Every 500K steps
- Best model auto-saved

---

## Phase 3: Full Human-Level Run (30+ hours, may span multiple sessions)

If Phase 2 shows continued improvement, extend to full human-level.

**Training config:**
| Parameter | Value |
|---|---|
| Steps | 12,500,000 |
| Envs | 4 |
| Total game frames | 200,000,000 |
| Expected time on GPU | ~33 hours |
| Expected score | ~13,000-20,000 |

**Success criteria:**
- Mean eval score > 13,513 (human expert)
- Mean eval score > 20,675 (Rainbow SOTA)

---

## Phase 4: Optimization & Beyond (if needed)

If human-level is not reached, apply these in order:

### 4.1 — Increase network capacity
- `hidden_dim`: 256 → 512
- Add attention layers (`--attention` flag, already implemented)
- Increases parameters from 3.5M to ~5.5M

### 4.2 — PBT hyperparameter search
- Use `pbt.py` to auto-tune lr, gamma, batch_size, target_update_freq
- Population of 8 members, 3-5 generations
- Requires ~3× compute budget

### 4.3 — ICM + Rainbow (curiosity-driven exploration)
- Already implemented in `agents/icm_rainbow.py`
- Helps explore sparse-reward regions (late-game high score zones)
- Adds an intrinsic reward bonus for novel states
- Tune `eta` (intrinsic reward scale) — start at 0.01

### 4.4 — Larger batch and buffer
- `batch_size`: 32 → 64 (n0 noise in gradients)
- `buffer_capacity`: 500K → 1M (holds more diverse experience)
- Requires ~2× more GPU memory (still well within 4GB)

---

## Appendix: Resource Budget Summary

| Phase | Steps | Game frames | GPU time | Expected score |
|---|---|---|---|---|
| 0 — Hyperparameter fix | Config change only | 0 | 0 min | Unlocks learning |
| 1 — Baseline validation | 500K | 8M | ~1.3 hr | ~2K-4K (3× gain) |
| 2 — Medium run | 3M | 48M | ~8 hr | ~5K-10K |
| 3 — Full human-level | 12.5M | 200M | ~33 hr | ~13K-20K |
| 4 — Optimization | Optional | — | varies | 20K+ |

**Total GPU time to human expert (Phase 1+2+3 if all succeed):** ~42 hours

**Expected score at each target:**
```
Current (588) ───► Phase 1 (~3K) ───► Phase 2 (~8K) ───► Phase 3 (~15K) ───► Beat human
    4.4% human        22% human         59% human          111% human
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Catastrophic forgetting with larger buffer | Medium | High | Monitor eval history; increase buffer if early peak then decline |
| v_max still too low for high-scoring episodes | Low | High | Monitor Q-values during training; expand v_max if q_value approaches v_max ceiling |
| GPU out-of-memory with 4 envs | Very low | High | Tested: 112 MB peak, 4GB available |
| Training extends beyond single session | High | Medium | Checkpoint auto-saved every 500K steps; resume with `--resume` |
| Score plateau before human-level | Medium | Medium | Phase 4 optimizations (attention, PBT, ICM) |
