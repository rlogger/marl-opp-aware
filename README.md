# Adaptive Opponent Modeling for Adversarial Co-Training in MARL

In adversarial co-training, each team faces an opponent whose strategy varies and
adapts; a best response that overfits to one opponent loses to strategies it has
not recently seen. This repo pursues the proposal's remedy — model the opponent's
latent strategy with **calibrated uncertainty** (Part 1) and **plan against it**
(Part 2) — and validates it in a controlled predator–prey game where the opponent
(the prey, `red`) draws a hidden strategy each episode and the controlled team
(the predators, `blue`) must infer it.

Everything is JAX (JaxMARL / MPE), runs on a laptop CPU, 3 seeds throughout.

- **Paper:** [`docs/intent_opponent_modeling.pdf`](docs/intent_opponent_modeling.pdf)
- **Site:** https://138-68-61-233.sslip.io

## Headline result

`blue` captures per episode against the varying `red` opponent (3 seeds × 300 eps):

| `blue` predator                       | captures / ep | vs. opponent-blind |
|---------------------------------------|:-------------:|:------------------:|
| opponent-blind (no strategy info)     | 2.68          | —                  |
| reactive, inferred belief             | 2.82          | +5%                |
| planner, flat belief (ablation)       | 3.07          | +15%               |
| oracle, true strategy (reactive)      | 4.05          | +51%               |
| **planner, inferred belief (Part 1+2)** | **4.31**    | **+61%**           |

The opponent's latent strategy is inferable from its behavior (encoder accuracy
0.37 → 0.97 over 3 → 25 observed steps); a point-estimate model is brittle (−47%
fed a wrong guess); and **planning against the uncertainty-aware opponent model is
the most robust response** — above even an oracle handed the true strategy. The
flat-belief ablation isolates the opponent inference as the source of the gain
(+40%).

## Repository map

### Flagship — adaptive opponent modeling (hidden-intent predator–prey)

The proposal's Part 1 (encode the opponent's latent strategy into a calibrated
belief) and Part 2 (sample that model inside a planner), instantiated so every
quantity is measurable against ground truth.

| file | role |
|------|------|
| `src/simple_tag_intent.py` | env: prey draws a hidden corner-intent each episode; predators don't see it. `reveal_to_pred` = oracle; `intent_belief_noise` = trains on soft beliefs (the uncertainty-aware Part-1 model). |
| `configs/alg/mappo_intent_{unaware,oracle,belief}.yaml` | the three co-training conditions (opponent-blind / oracle / belief). |
| `src/part2_intent_eval.py` | **Part 1**: encoder inferring intent from the prey's first-*k* steps (accuracy + posterior entropy), and the unaware/guess/inferred/oracle capture ladder. → `plots/part2_intent_eval.png`. |
| `src/part2_planner.py` | **Part 2**: belief-conditioned Monte-Carlo planner — samples the opponent's moves under a strategy drawn from the belief. → `plots/part2_planner.png`. |
| `src/intent_demo.py` | demo: belief predators infer the corner and converge. → `plots/demo_intent.gif`. |
| `src/part3_learned_planner.py` | exploratory: the planner through a *learned* dynamics model (feasible but model-limited; not in the paper). |

### Shared infrastructure

| file | role |
|------|------|
| `src/mappo_teams_mlp.py` | two-team MAPPO trainer (CTDE, centralized critic). Builds the intent / resource / static envs from config. |
| `src/iql_teams_mlp.py`, `src/iql_teams_oa_mlp.py` | two-team IQL / opponent-aware IQL trainers (MLP). |
| `src/generate_trajectory_dataset_resources.py` | greedy-rollout helpers, `ActorLogits`, checkpoint loaders (reused by the intent eval). |
| `src/train_traj_vae.py`, `src/planning_eval.py` | trajectory VAE; MC planner used in Exp 3. |

### Build-up experiments (Exp 1–5, documented on the site)

Superseded by the flagship but kept for the record. Exp 1 RNN A/B
(`iql_teams*.py`, `compare_plots.py`); Exp 2 static shaping + VAE
(`simple_tag_static.py`, `exp2_behavior_mining.py`, `verify_*.py`); Exp 3 cross-play
tournament (`tournament.py`, `plot_exp3.py`); Exp 4 resources
(`simple_tag_resources.py`, `tournament_resources.py`, `exp4_vae_separation.py`,
`plot_exp4.py`); Exp 5 specialists / brittleness / latent-BC
(`exp5_specialist_strategy.py`, `exp5b_encoder_length.py`,
`brittleness_crossplay.py`, `part1_latent_bc.py`). `smoke_test_*.py` are the env
unit tests.

## Reproduce the flagship

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e JaxMARL/ "flax==0.10.2" hydra-core flashbax wandb matplotlib distrax optax scikit-learn

# co-train the three conditions (MAPPO + CTDE, 3 seeds, ~2 min each)
python src/mappo_teams_mlp.py alg=mappo_intent_unaware NUM_SEEDS=3
python src/mappo_teams_mlp.py alg=mappo_intent_oracle  NUM_SEEDS=3
python src/mappo_teams_mlp.py alg=mappo_intent_belief  NUM_SEEDS=3

# Part 1 — encoder + the unaware/inferred/oracle ladder  -> plots/part2_intent_eval.png
python src/part2_intent_eval.py
# Part 2 — belief-conditioned planner                    -> plots/part2_planner.png
python src/part2_planner.py
# demo                                                    -> plots/demo_intent.gif
python src/intent_demo.py
```

Trained checkpoints land in `logs/` and are **not** tracked in git — they are
regenerated deterministically by the training commands above. Figures live in
`plots/`; the site source is in `site/`; the papers (this study plus the Exp 4/5
write-ups) are in `docs/`.

## Setup

Three `blue` predators vs one `red` prey on JaxMARL's `MPE_simple_tag_v3` (five
discrete actions, 25-step episodes, two fixed obstacles). Predators are slowed
(`PRED_MAX_SPEED=0.6`) so the prey can express its strategy. Each prey draws a
hidden intent `z ∈ {0,1,2,3}` (a corner it is rewarded for haunting); the prey
sees `z`, the predators do not. MAPPO: 32 envs × 128-step rollouts, γ=0.99,
λ=0.95, clip 0.2, 4 epochs, hidden 128.
