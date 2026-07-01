# Contributing

This is research code. The priority is reproducibility, clear provenance, and
small changes that preserve the interpretation of reported results.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

For full experiments, install `requirements-full.txt` and an editable JaxMARL
checkout as described in `docs/REPRODUCIBILITY.md`.

## Before Opening A PR

Run:

```bash
python -m pytest tests
python -m compileall mopa tests
git diff --check
```

If regenerated experiment artifacts are available, also run:

```bash
mopa-check-results
```

## Research-Code Guidelines

- Keep train/validation splits at the episode level unless a script explicitly
  documents a different protocol.
- Do not use future trajectory steps to construct conditioning latents for
  predictions at earlier steps.
- Keep raw result artifacts under `logs/`; do not commit checkpoints or raw
  training logs.
- When adding a paper-facing result, add it to `docs/RESULTS_MANIFEST.md` with
  the figure, script, raw artifact, and caveat.
- Prefer small, auditable utilities in `mopa/` over copying protocol logic across
  scripts.
