# Reproducibility Checklist

This repo has two verification layers.

## Fast Protocol Tests

These tests do not need JAX or checkpoints. They check the leakage-sensitive
utilities that the experiments share.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest tests
python -m compileall mopa tests
git diff --check
```

Equivalent shortcut:

```bash
make check
```

Covered invariants:

- Future steps are masked in fixed encoder windows.
- Occupancy rows are normalized.
- BC samples skip auto-reset boundaries where velocity would leak across
  episodes.
- Episode-level train/validation splits keep all timesteps from an episode in
  one fold.
- Publication result thresholds parse the raw `.npz` artifacts correctly.

## Result Threshold Checks

After regenerating the experiment artifacts under `logs/MPE_simple_tag_v3`, run:

```bash
mopa-check-results
```

The checker expects:

- `mopa_bc_vs_mappo.npz`: BC recovers at least 75% of MAPPO's edge over random
  and has at least 75% held-out action match.
- `mopa_bc_latent_sweep.npz`: the best latent-conditioned BC beats vanilla by at
  least one action-accuracy point, reaches the oracle accuracy band, and has a
  latent probe of at least 0.75.
- `mopa_bc_latent_deploy.npz`: the oracle placement recovers at least 70% of the
  pooled-BC to MAPPO deployed-captures gap, and the unsupervised latent does not
  materially hurt deployed captures.

Use `mopa-check-results --allow-missing` only for CI or partial reruns where some
raw artifacts have not been regenerated yet.

## Full Experiment Reruns

The full experiments need the normal research environment: JAX, Flax, Optax,
scikit-learn, JaxMARL, and regenerated MAPPO/resource checkpoints.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-full.txt
pip install -e ../JaxMARL/
pip install -e .

python -m mopa.experiments.bc_vs_mappo
python -m mopa.experiments.bc_latent_sweep
python -m mopa.experiments.bc_latent_deploy
mopa-check-results
```

The raw metrics are intentionally ignored in git under `logs/`; figures and the
presentation are regenerated from those artifacts.

## Presentation Rebuild

The presentation source is `scripts/build_deck.mjs` and uses OpenAI's
`@oai/artifact-tool` presentation API. In Codex, initialize an artifact-tool
workspace first, then run the script from an environment where
`@oai/artifact-tool` is resolvable:

```bash
node "$SKILL_DIR/container_tools/setup_artifact_tool_workspace.mjs" \
  --workspace /tmp/marl-opp-aware-deck
ARTIFACT_TOOL_WORKSPACE=/tmp/marl-opp-aware-deck node scripts/build_deck.mjs
```

It writes `marl_opp_aware_results.pptx` and rendered QA previews under
`marl_opp_aware_results/`.

## Archival Release

Before a paper submission or camera-ready release:

- tag the exact git commit used for the paper;
- archive the repo through Zenodo or a comparable archive;
- archive raw `logs/` artifacts/checkpoints separately if reviewers need
  byte-for-byte reruns;
- update `CITATION.cff` and `.zenodo.json` with the paper DOI/arXiv and archive
  DOI.
