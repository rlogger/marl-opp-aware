# Paper Artifact Guide

This repository is intended to accompany a paper on adaptive opponent modeling
for adversarial co-training in multi-agent reinforcement learning.

## What This Artifact Supports

The paper-ready claim is:

> In a controlled hidden-intent predator-prey task, maintaining a calibrated
> belief over the opponent's latent strategy and sampling that belief inside a
> planner is more robust than reacting to a point estimate. A predictive
> self-supervised encoder can recover a useful opponent belief without intent
> labels.

The current artifact supports that claim with:

- source code for the hidden-intent environment, MAPPO training, belief
  evaluation, and planners;
- reusable `mopa/` modules for feature extraction, self-supervised encoders,
  behaviour cloning, and result checks;
- committed figures and papers under `plots/` and `docs/`;
- fast protocol tests under `tests/`;
- result-threshold checks for regenerated raw `.npz` artifacts.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `mopa/` | Reusable package code for current experiments and checks. |
| `mopa/experiments/` | Packaged, rerunnable experiment entry points. |
| `src/` | Historical and flagship research scripts, including environment and MAPPO/planner implementations. |
| `configs/` | Hydra configs for training conditions. |
| `plots/` | Committed figure artifacts used by the paper and presentation. |
| `docs/` | Paper PDFs/TeX, methods, literature review, presentation notes, and reproducibility docs. |
| `tests/` | Fast tests for leakage-sensitive protocol utilities. |
| `scripts/build_deck.mjs` | Rebuilds the presentation from committed figures. |

## Artifact Claims And Caveats

### Supported Claims

- Hidden intent is inferable from motion: encoder accuracy rises from `0.37` to
  `0.97` while posterior entropy falls from `1.35` to `0.03` nats.
- Point estimates are brittle: a confident wrong intent reaches `1.42`
  captures/episode, below the opponent-blind baseline `2.68`.
- Belief-conditioned planning is the strongest current result: `4.31`
  captures/episode versus `3.07` for the flat-belief planner and `2.82` for a
  reactive belief policy.
- JEPA is the better hidden-intent encoder: probe `0.89` versus VAE `0.53`,
  with no intent labels during encoder training.
- The resource-axis BC experiments show downstream value for strategy
  information, but the current unsupervised resource latent is not yet a
  paper-level deployed-captures win.

### Scope Boundaries

- The opponent intent is static within each episode; adaptive and switching
  opponents are future work.
- The main planner is simulator-based; learned-model planning is included as an
  honest negative/boundary result.
- "Beats oracle" means "matches or slightly exceeds a reactive oracle policy,"
  not an oracle planner.
- The resource-layout latent is linearly decodable but does not form clean
  unsupervised clusters yet.
- Modern model-based MARL baselines such as HOP, MAZero, MAMBA, MARIE, MATWM,
  MBOM, and AORPO are not yet implemented.

## Review Workflow

For a lightweight artifact review:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pytest
python -m pytest tests
python -m compileall mopa tests
```

For a full result reproduction, install the experiment environment, regenerate
checkpoints and raw metrics, then run:

```bash
mopa-check-results
```

See `docs/REPRODUCIBILITY.md` for exact commands and expected artifacts.

## Data And Checkpoint Policy

Raw training logs, checkpoints, and regenerated `.npz` metrics live under
`logs/` and are intentionally ignored by git. They are reproducible from the
training/evaluation scripts but should be archived alongside a paper release
using Zenodo, institutional storage, or a release asset if exact byte-for-byte
review is required.

The committed figures in `plots/` are stable paper/presentation artifacts. The
raw result checker is the guardrail that should be run after regenerating the
ignored metrics.

## Hardware And Runtime Notes

The flagship experiments were designed for laptop CPU-scale JAX runs. MAPPO
training and full deployment evaluations need the complete JAX/JaxMARL research
environment. The fast unit tests do not need JAX or checkpoints.

## Recommended Pre-Submission Checklist

- Run `python -m pytest tests`.
- Run `python -m compileall mopa tests`.
- Regenerate the raw `.npz` artifacts and run `mopa-check-results`.
- Rebuild the presentation with `scripts/build_deck.mjs`.
- Archive logs/checkpoints/raw metrics externally and add the archive DOI or URL
  to `README.md`, `CITATION.cff`, and `.zenodo.json`.
- Add paper DOI/arXiv information once available.
