# Opponent-Aware IQL in a Predator–Prey MPE

Two-team independent Q-learning on JaxMARL's `MPE_simple_tag_v3` (three predators against one prey), extended with an opponent-action auxiliary head and a tree-expanded Monte-Carlo planner that reuses the head as a one-step opponent model.

The repository contains three experiments:

1. **Experiment 3** (primary). MLP-IQL, OA-IQL-MLP (shared trunk, Q head and opponent-action head), and a tree-expanded Monte-Carlo planner. Evaluated in a 3×3 paired-seed cross-play tournament (three seeds per variant).
2. **Experiment 2.** Static obstacles + parameterised CW/CCW prey shaping. Three seeds per variant. Behavior-level mining of the learned prey policy.
3. **Experiment 1** (archived). Single-seed RNN A/B between baseline IQL and OA-IQL. Superseded by Experiment 3.

---

## Headline numbers

**Experiment 3 — captures per episode (mean ± std, 3 seeds × 100 eval eps).**

|                      | prey IQL-grdy | prey OA-grdy | prey OA-plan |
|----------------------|---------------|--------------|--------------|
| **pred IQL-greedy**  | 3.41 ± 0.53   | 3.64 ± 0.39  | 1.57 ± 0.15  |
| **pred OA-greedy**   | 3.39 ± 1.00   | 3.01 ± 0.30  | 1.15 ± 0.25  |
| **pred OA-plan**     | 2.27 ± 0.59   | 2.80 ± 0.49  | 1.07 ± 0.30  |

OA-plan prey is uniformly the lowest column. OA-plan pred is the lowest-mean row. The mechanism is the opp-head accuracy gap: prey 0.674 ± 0.032 vs predator 0.607 ± 0.021 (chance = 0.20), traceable to the supervision asymmetry — prey classifies three predator actions per step, predator classifies one.

**Experiment 2 — captures per episode under directional shaping.**

| Run               | Captures / ep   |
|-------------------|-----------------|
| static_baseline   | 2.22 ± 0.60     |
| static_ccw        | 2.62 ± 0.28     |
| static_cw         | 2.82 ± 0.21     |

Behavior mining shows a sign-correct angular bias in the learned prey policy (cw −0.011 rad/step, ccw +0.009 rad/step). The shaping concentrates prey position density into orbital bands; the narrower trajectory distribution is exploited by the coordinated predators.

---

## Layout

```
src/        # training, planning, evaluation, plotting
configs/    # Hydra configs per experiment + algorithm
logs/       # safetensors params + metrics npz, by experiment
plots/      # figures + rollout GIFs
```

---

## Reproducing

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e JaxMARL/ "flax==0.10.2" hydra-core flashbax wandb matplotlib

# Experiment 3 (~5 min on CPU)
python src/iql_teams_mlp.py    alg=ql_teams_mlp_simple_tag    NUM_SEEDS=3
python src/iql_teams_oa_mlp.py alg=ql_teams_oa_mlp_simple_tag NUM_SEEDS=3
python src/tournament.py --seeds 3 --eps 100 --K 5 --H 3
python src/plot_exp3.py

# Experiment 2 (~24 min, RNN trainings)
python src/iql_teams.py alg=ql_teams_static_baseline NUM_SEEDS=3
python src/iql_teams.py alg=ql_teams_static_cw       NUM_SEEDS=3
python src/iql_teams.py alg=ql_teams_static_ccw      NUM_SEEDS=3
python src/exp2_behavior_mining.py
```

---

## Hyperparameters

`2 × 10⁶` env-steps per training run · 8 parallel envs · 25-step episodes · γ = 0.9 · ε 1.0 → 0.05 over 10 % of training · LR 0.005 with linear decay · target update every 200 steps (τ = 1) · buffer 5000 / batch 32 · MLP hidden 128 (Exp 3) / RNN hidden 64 (Exp 1, 2) · `OPP_AUX_COEF = 0.5` · planner K = 5, H = 3, γ = 0.9.
