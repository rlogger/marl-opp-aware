# Opponent-Aware IQL in a Predator–Prey MPE

Two-team adversarial RL on JaxMARL's `MPE_simple_tag_v3` — three predators chase one prey. The main thing is a two-team IQL split (separate Q-nets per team, shared weights within team) plus an opponent-action auxiliary head that can be reused as a one-step opponent model for MC planning at eval. Exp 4 adds MAPPO and a resource-collection env. Everything is JAX, everything runs on CPU in a few minutes.

Four experiments, roughly in order of relevance:

- **Exp 4** — Resource-augmented env (`SimpleTagResourcesMPE`): 4 prey-visible resources, three placement modes (circle / corners / random). Trains IQL, OA-IQL, and two-team MAPPO. MAPPO gets ~3x the predator return of either IQL variant. Trajectory datasets for a downstream VAE.
- **Exp 3** — 3x3 cross-play tournament on the base env (IQL-greedy / OA-greedy / OA-plan). The opp-action head disproportionately helps the prey because it classifies 3 actions per step vs 1. This is the main result for the opponent-modeling angle.
- **Exp 2** — Static obstacles with CW/CCW directional shaping. Behavior mining shows sign-correct angular bias; the narrower trajectory distribution gets exploited by predators. VAE clustering fails to separate policies per-trajectory (negative result).
- **Exp 1** — Archived. Single-seed RNN A/B, superseded by Exp 3.

---

## Results

Exp 4, random placement, 3 seeds:

| Algorithm | Pred return     | Prey return       | Pred resources | Prey resources |
|-----------|-----------------|-------------------|----------------|----------------|
| IQL       | +33.5 ± 1.4    | −37.0 ± 2.4       | 0.415          | 0.444          |
| OA-IQL    | +35.5 ± 1.8    | −48.8 ± 4.9       | 0.414          | 0.387          |
| MAPPO     | +110.5 ± 12.3  | −113.7 ± 5.7      | 0.460          | 0.468          |

The MAPPO gap comes from the centralized critic coordinating pursuit. Resource collection is about the same across all three, so the return difference is entirely tag/evasion.

Exp 3, captures/ep (3 seeds x 100 eval eps):

|                      | prey IQL-grdy | prey OA-grdy | prey OA-plan |
|----------------------|---------------|--------------|--------------|
| **pred IQL-greedy**  | 3.41 ± 0.53   | 3.64 ± 0.39  | 1.57 ± 0.15  |
| **pred OA-greedy**   | 3.39 ± 1.00   | 3.01 ± 0.30  | 1.15 ± 0.25  |
| **pred OA-plan**     | 2.27 ± 0.59   | 2.80 ± 0.49  | 1.07 ± 0.30  |

OA-plan prey is the hardest to catch regardless of predator variant. The prey's opp-head accuracy is 0.674 vs pred's 0.607 (chance 0.20) — 3x the classification signal per step.

---

## How MAPPO works here

Not cooperative. Two separate actor-critic pairs (pred team / prey team), each with its own reward stream (+10 / -10 per capture). Within the pred team, the 3 predators share weights. The critic is centralized (concatenated world state from all agents) but per-team — the pred critic learns pred values, prey critic learns prey values. Standard CTDE setup, just applied adversarially.

IQL / OA-IQL use per-team `TrainState`s with a shared flashbax replay buffer. OA-IQL's auxiliary head predicts opponent actions (CE loss, coef 0.5); at eval the OA-plan variant feeds it into a K=5, H=3 MC lookahead.

---

## Layout

```
src/        training, planning, evaluation, plotting
configs/    Hydra YAML per experiment
logs/       checkpoints (safetensors) + metrics (npz)
plots/      figures and rollout GIFs
site/       project website
docs/       LaTeX writeup
```

The main files: `iql_teams_mlp.py`, `iql_teams_oa_mlp.py`, `mappo_teams_mlp.py` (trainers); `simple_tag_resources.py` (resource env); `tournament.py` / `tournament_resources.py` (cross-play eval); `visualize_rollout_resources.py` (GIF rendering).

---

## Reproducing

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e JaxMARL/ "flax==0.10.2" hydra-core flashbax wandb matplotlib distrax optax

# Exp 4 — resource env (~7 min on CPU)
python src/smoke_test_resources.py
python src/iql_teams_mlp.py    alg=ql_teams_resources_random    NUM_SEEDS=3
python src/iql_teams_oa_mlp.py alg=ql_teams_oa_resources_random NUM_SEEDS=3
python src/mappo_teams_mlp.py  alg=mappo_teams_resources_random  NUM_SEEDS=3
python src/plot_exp4.py
python src/visualize_rollout_resources.py --all

# Exp 3 — tournament (~5 min)
python src/iql_teams_mlp.py    alg=ql_teams_mlp_simple_tag    NUM_SEEDS=3
python src/iql_teams_oa_mlp.py alg=ql_teams_oa_mlp_simple_tag NUM_SEEDS=3
python src/tournament.py --seeds 3 --eps 100 --K 5 --H 3
python src/plot_exp3.py

# Exp 2 — shaping (~24 min, RNN)
python src/smoke_test_static.py
python src/iql_teams.py alg=ql_teams_static_baseline NUM_SEEDS=3
python src/iql_teams.py alg=ql_teams_static_cw       NUM_SEEDS=3
python src/iql_teams.py alg=ql_teams_static_ccw      NUM_SEEDS=3
python src/exp2_behavior_mining.py
```

## Hyperparameters

IQL / OA-IQL: 2M steps, 8 envs, 25-step eps, γ=0.9, ε 1→0.05 over 10%, LR 0.005 linear decay, target update every 200 (hard), buffer 5k / batch 32, hidden 128, opp aux coef 0.5, planner K=5 H=3.

MAPPO: 2M steps, 32 envs, 128-step rollouts, γ=0.99 λ=0.95, LR 3e-4 linear anneal, clip 0.2, ent 0.01, vf 0.5, 4 epochs / 4 minibatches, grad norm 0.5, hidden 128.
