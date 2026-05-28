# Opponent-Aware IQL in a Predator–Prey MPE

Two-team adversarial RL on JaxMARL's `MPE_simple_tag_v3` (three predators vs one prey). IQL, opponent-aware IQL, and MAPPO with an opponent-action auxiliary head and tree-expanded Monte-Carlo planning. Built in JAX, runs on a laptop CPU.

## Experiments

1. **Experiment 4** (latest). Resource-augmented predator-prey (`SimpleTagResourcesMPE`). 4 collectable resources, prey-visible only. Three placement modes (circle, corners, random). IQL vs OA-IQL vs MAPPO — MAPPO outperforms IQL ~3× on predator returns. Trajectory dataset generation for downstream VAE analysis.
2. **Experiment 3**. MLP-IQL, OA-IQL (shared trunk + opponent-action head), and tree-expanded MC planner. 3×3 cross-play tournament, 3 seeds per variant.
3. **Experiment 2.** Static obstacles + CW/CCW directional prey shaping. Behavior mining of learned prey policy. Per-trajectory VAE clustering (negative result: latent encodes geometry, not policy).
4. **Experiment 1** (archived). Single-seed RNN A/B. Superseded by Experiment 3.

---

## Headline numbers

**Experiment 4 — training results, random placement (mean ± std, 3 seeds).**

| Algorithm | Pred return     | Prey return       | Pred resources | Prey resources |
|-----------|-----------------|-------------------|----------------|----------------|
| IQL       | +33.5 ± 1.4    | −37.0 ± 2.4       | 0.415          | 0.444          |
| OA-IQL    | +35.5 ± 1.8    | −48.8 ± 4.9       | 0.414          | 0.387          |
| **MAPPO** | **+110.5 ± 12.3** | **−113.7 ± 5.7** | 0.460          | 0.468          |

MAPPO's centralized critic coordinates predator pursuit (~3× IQL). Resource collection comparable across algorithms — the gap is tag/evasion, not resources.

**Experiment 3 — captures per episode (mean ± std, 3 seeds × 100 eval eps).**

|                      | prey IQL-grdy | prey OA-grdy | prey OA-plan |
|----------------------|---------------|--------------|--------------|
| **pred IQL-greedy**  | 3.41 ± 0.53   | 3.64 ± 0.39  | 1.57 ± 0.15  |
| **pred OA-greedy**   | 3.39 ± 1.00   | 3.01 ± 0.30  | 1.15 ± 0.25  |
| **pred OA-plan**     | 2.27 ± 0.59   | 2.80 ± 0.49  | 1.07 ± 0.30  |

OA-plan prey uniformly lowest column. Mechanism: opp-head accuracy gap (prey 0.674 vs pred 0.607, chance 0.20) from supervision asymmetry — prey classifies 3 predator actions per step.

---

## Architecture

**MAPPO**: two-team (not cooperative). Per-team actor-critic pair with parameter sharing within team. Centralized critic sees concatenated world state; individual per-agent rewards (pred +10, prey −10 per capture). CTDE — centralized training, decentralized execution. 32 parallel envs, 128-step rollouts, PPO clipped surrogate + GAE.

**IQL / OA-IQL**: per-team `TrainState`, shared flashbax replay buffer. OA-IQL adds an auxiliary head predicting opponent actions (CE loss, coef 0.5). At eval, OA-plan reuses the head as a one-step opponent model for K=5, H=3 MC lookahead.

---

## Layout

```
src/        # training, planning, evaluation, plotting
configs/    # Hydra configs per experiment + algorithm
logs/       # safetensors params + metrics npz
plots/      # figures + rollout GIFs
site/       # project website (deployed to droplet)
docs/       # LaTeX writeup (exp4_results.pdf)
```

Key source files:
- `iql_teams_mlp.py` — two-team IQL trainer (MLP)
- `iql_teams_oa_mlp.py` — opponent-aware IQL trainer
- `mappo_teams_mlp.py` — two-team MAPPO trainer
- `simple_tag_resources.py` — resource-augmented environment
- `visualize_rollout_resources.py` — GIF rendering
- `tournament.py` / `tournament_resources.py` — cross-play evaluation
- `plot_exp3.py` / `plot_exp4.py` — training curve plots

---

## Reproducing

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e JaxMARL/ "flax==0.10.2" hydra-core flashbax wandb matplotlib distrax optax

# Experiment 4 — resource collection (~7 min training on CPU)
python src/smoke_test_resources.py
python src/iql_teams_mlp.py    alg=ql_teams_resources_random  NUM_SEEDS=3
python src/iql_teams_oa_mlp.py alg=ql_teams_oa_resources_random NUM_SEEDS=3
python src/mappo_teams_mlp.py  alg=mappo_teams_resources_random NUM_SEEDS=3
python src/plot_exp4.py
python src/visualize_rollout_resources.py --all

# Experiment 3 (~5 min training + ~50 s tournament)
python src/iql_teams_mlp.py    alg=ql_teams_mlp_simple_tag    NUM_SEEDS=3
python src/iql_teams_oa_mlp.py alg=ql_teams_oa_mlp_simple_tag NUM_SEEDS=3
python src/tournament.py --seeds 3 --eps 100 --K 5 --H 3
python src/plot_exp3.py

# Experiment 2 (~24 min, RNN)
python src/smoke_test_static.py
python src/iql_teams.py alg=ql_teams_static_baseline NUM_SEEDS=3
python src/iql_teams.py alg=ql_teams_static_cw       NUM_SEEDS=3
python src/iql_teams.py alg=ql_teams_static_ccw      NUM_SEEDS=3
python src/exp2_behavior_mining.py
```

---

## Hyperparameters

**IQL / OA-IQL** (off-policy): 2M env-steps · 8 parallel envs · 25-step episodes · γ = 0.9 · ε 1.0 → 0.05 over 10% · LR 0.005 linear decay · target update every 200 steps (τ = 1) · buffer 5000 / batch 32 · MLP hidden 128 · `OPP_AUX_COEF = 0.5` · planner K=5, H=3.

**MAPPO** (on-policy): 2M env-steps · 32 parallel envs · 128-step rollouts · γ = 0.99, λ = 0.95 · LR 0.0003 linear anneal · PPO clip 0.2 · entropy coef 0.01 · value coef 0.5 · 4 epochs, 4 minibatches · max grad norm 0.5 · MLP hidden 128.
