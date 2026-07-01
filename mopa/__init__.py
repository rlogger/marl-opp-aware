"""mopa — Modeling OPponent Agents.

Reusable core for the paper's experiments: opponent-trajectory encoders
(VAE vs JEPA), specialist-rollout data loading, and latent-conditioned
behaviour cloning, on the predator-prey-resources MARL testbed.

The legacy single-file experiments live in src/; this package wraps the
checkpoint/rollout machinery there (see mopa.legacy) and provides the
clean, seed-averaged, leakage-safe implementations used going forward.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
