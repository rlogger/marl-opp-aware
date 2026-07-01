# Adaptive Opponent Modeling for Adversarial Co-Training in MARL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/rlogger/marl-opp-aware/actions/workflows/ci.yml/badge.svg)](https://github.com/rlogger/marl-opp-aware/actions/workflows/ci.yml)
[![JAX](https://img.shields.io/badge/JAX-0.4-blue)](https://github.com/jax-ml/jax)
[![Site](https://img.shields.io/badge/results-site-0d6e7a)](https://138-68-61-233.sslip.io)

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
- **Minimal talk deck:** [`marl_opp_aware_results.pptx`](marl_opp_aware_results.pptx)
- **Paper artifact guide:** [`docs/PAPER_ARTIFACT.md`](docs/PAPER_ARTIFACT.md)
- **Results manifest:** [`docs/RESULTS_MANIFEST.md`](docs/RESULTS_MANIFEST.md)
- **Reproducibility:** [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)

## Quick start

For lightweight artifact review (no JAX/JaxMARL checkpoints required):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
make check
```

For the full research environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-full.txt
# Install JaxMARL as an editable sibling checkout, then:
pip install -e .
```

The full experiments also need regenerated checkpoints under `logs/`, which are
not tracked in git. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Headline result

`blue` captures per episode against the varying `red` opponent (3 seeds × 300 eps):

| `blue` predator                       | captures / ep | vs. opponent-blind |
|---------------------------------------|:-------------:|:------------------:|
| opponent-blind (no strategy info)     | 2.68          | —                  |
| reactive, hard-inferred intent (@k=8) | 2.56          | −5%                |
| reactive belief (uncertainty-aware)   | 2.82          | +5%                |
| planner, flat belief (ablation)       | 3.07          | +15%               |
| oracle, true strategy (reactive)      | 4.05          | +51%               |
| **planner, inferred belief (Part 1+2)** | **4.31**    | **+61%**           |

The opponent's latent strategy is inferable from its behavior (encoder accuracy
0.37 → 0.97 over 3 → 25 observed steps); a point-estimate model is brittle (−47%
fed a wrong guess, and even an honestly *inferred* hard guess at a fixed step
loses, −5%); and **planning against the uncertainty-aware opponent model is
the most robust response** — above even an oracle handed the true strategy. The
flat-belief ablation isolates the opponent inference as the source of the gain
(+40%).

## The `mopa` package (new experiments live here)

The reusable core is packaged as `mopa/` (installable: `pip install -e .`), and
the current meeting deliverables run as modules:

```bash
# VAE vs JEPA strategy latents on circle-vs-corners (scatter + context sweep)
python -m mopa.experiments.latent_resources --algorithm mappo --team prey
python -m mopa.experiments.latent_resources --algorithm mappo --team pred

# predator behaviour cloning: pi(a|s) vs pi(a|s,z) (episode-level split)
python -m mopa.experiments.bc_latent --algorithm mappo

# deployed-captures checks used in the presentation/paper artifact
python -m mopa.experiments.bc_vs_mappo
python -m mopa.experiments.bc_latent_sweep
python -m mopa.experiments.bc_latent_deploy
mopa-check-results
```

`mopa.data` rolls out the placement specialists in their native envs with
ABSOLUTE coordinates (the placement signal is *where* the prey routes; the old
init-conditioned featurisation removed it, which is why exp4/exp5's single-
trajectory probes sat at chance). `mopa.encoders` is the validated VAE/JEPA
pair with multi-seed evaluation; `mopa.bc` is leakage-safe BC (episode-level
splits, conditioning never sees past the predicted step).

For verification, see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md):
fast tests cover masking/splitting/sample construction, and `mopa-check-results`
checks regenerated raw `.npz` metrics against paper-facing thresholds.

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
pip install -r requirements-full.txt
pip install -e ../JaxMARL/
pip install -e .

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
`plots/`; the papers (this study plus the Exp 4/5
write-ups) are in `docs/`.

## Publication artifact status

This repository is suitable as a paper code artifact for the current controlled
study: it includes source, figures, papers, tests, citation metadata, environment
files, CI, and a result-check harness. The remaining publication upgrades are
scientific rather than packaging-related: add adaptive/switching opponents, an
oracle planner, and external baselines such as HOP, MAZero/MAMBA/MARIE/MATWM,
MBOM, and AORPO.

## Setup

Three `blue` predators vs one `red` prey on JaxMARL's `MPE_simple_tag_v3` (five
discrete actions, 25-step episodes, two fixed obstacles). Predators are slowed
(`PRED_MAX_SPEED=0.6`) so the prey can express its strategy. Each prey draws a
hidden intent `z ∈ {0,1,2,3}` (a corner it is rewarded for haunting); the prey
sees `z`, the predators do not. MAPPO: 32 envs × 128-step rollouts, γ=0.99,
λ=0.95, clip 0.2, 4 epochs, hidden 128.

## Citing

If you use this code or results, please cite (see `CITATION.cff`; a Zenodo DOI
is minted per release):

```bibtex
@software{singh2026oppaware,
  author  = {Singh, Rajdeep},
  title   = {Adaptive Opponent Modeling for Adversarial Co-Training in
             Multi-Agent Reinforcement Learning},
  year    = {2026},
  url     = {https://github.com/rlogger/marl-opp-aware},
  version = {0.1.0}
}
```

## License

MIT — see [LICENSE](LICENSE).
