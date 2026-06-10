"""Specialist-rollout datasets for the circle-vs-corners resource task.

Rolls out the placement-specialist checkpoints (circle / corners), each in its
native environment, and returns ABSOLUTE-coordinate trajectories. Absolute
coordinates are deliberate: the placement signal lives in WHERE the prey goes
(circle radius 0.6 vs corners radius ~1.13); the earlier init-conditioned
featurisation (exp4/exp5) removed exactly that signal, which is why its raw
probe sat at chance.

Episode structure: rollouts are NUM_STEPS=50 with an auto-reset at t=25, so
steps [0, 25) are a complete first episode whose start state pairs with the
trajectory; we expose that slice for encoder windows and mark the reset
boundary for BC sample filtering.
"""
import numpy as np
import jax

from mopa import legacy  # noqa: F401  (sys.path bootstrap)
import generate_trajectory_dataset_resources as G

PLACEMENTS = ("circle", "corners")
EP_LEN = 25                       # auto-reset boundary inside the 50-step rollout


def specialist_dataset(algorithm="mappo", ckpt_suffix="_wp", n_eps=200,
                       ckpt_seeds=(0, 1, 2), pred_max_speed=0.6,
                       pred_accel=1.5, collect_reward=10.0, rng0=0):
    """Roll out both specialists in their native envs; return pooled arrays.

    Returns dict with
      prey_pos  (N, 51, 2)  absolute prey positions
      pred_pos  (N, 51, 3, 2) absolute predator positions
      prey_act  (N, 50)     prey actions
      pred_act  (N, 50, 3)  predator actions
      label     (N,)        0 = circle specialist, 1 = corners specialist
      ckpt_seed (N,)        which checkpoint seed produced the episode
    """
    G.PRED_MAX_SPEED = pred_max_speed
    G.PRED_ACCEL = pred_accel
    G.COLLECT_REWARD = collect_reward

    rows = {k: [] for k in ("prey_pos", "pred_pos", "prey_act", "pred_act",
                            "label", "ckpt_seed")}
    for lab, placement in enumerate(PLACEMENTS):
        for s in ckpt_seeds:
            d = G.rollout_one_checkpoint(
                algorithm, placement + ckpt_suffix, s, n_eps,
                eval_placement=placement,
                rng_key=jax.random.PRNGKey(rng0 + 100 * lab + s))
            n = len(d["positions"])
            rows["prey_pos"].append(d["positions"])
            rows["pred_pos"].append(d["pred_positions"])
            rows["prey_act"].append(d["actions"])
            rows["pred_act"].append(d["pred_actions"])
            rows["label"].append(np.full(n, lab, np.int32))
            rows["ckpt_seed"].append(np.full(n, s, np.int32))
    return {k: np.concatenate(v).astype(np.float32 if "pos" in k else np.int32)
            for k, v in rows.items()}


def window(pos, k, kmax=EP_LEN):
    """First-episode fixed-size window: first k steps kept, steps [k, kmax)
    zero-masked (the I-JEPA masked-future convention). pos is (N, T+1, 2) or
    (N, T+1, A, 2); returns (N, kmax*D) flattened float32."""
    ep = pos[:, :kmax]                                  # (N, kmax, ...)
    flat = ep.reshape(len(ep), kmax, -1).copy()
    flat[:, k:] = 0.0
    return flat.reshape(len(ep), -1).astype(np.float32)


def standardize(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-6
    return (X - mu) / sd, mu, sd
