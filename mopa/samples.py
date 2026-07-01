"""Behaviour-cloning sample construction for predator policies."""

from __future__ import annotations

import numpy as np

from mopa.features import EP_LEN

N_PREDATORS = 3
PREDATOR_FEATURE_DIM = 19


def valid_bc_steps(t0: int, t1: int, ep_len: int = EP_LEN) -> tuple[int, ...]:
    """Steps with a valid one-step velocity feature.

    Auto-reset boundaries are excluded because ``pos[t] - pos[t-1]`` would span
    two episodes there. Step zero is excluded for the same reason.
    """

    if t0 < 0 or t1 <= t0:
        raise ValueError(f"invalid step range [{t0}, {t1})")
    return tuple(t for t in range(max(1, t0), t1) if t % ep_len != 0)


def predator_state_features(
    preds: np.ndarray,
    prey: np.ndarray,
    prev_preds: np.ndarray,
    prev_prey: np.ndarray,
) -> np.ndarray:
    """Build per-predator BC features for one timestep.

    Returns ``(N, 3, 19)`` with positions of all agents, a one-step velocity
    proxy, and a predator-id one-hot. This is the canonical feature order used
    by all BC experiments.
    """

    if preds.ndim != 3 or preds.shape[1:] != (N_PREDATORS, 2):
        raise ValueError("preds must have shape (N, 3, 2)")
    if prev_preds.shape != preds.shape:
        raise ValueError("prev_preds must match preds")
    if prey.shape != (len(preds), 2) or prev_prey.shape != prey.shape:
        raise ValueError("prey and prev_prey must have shape (N, 2)")

    n = len(prey)
    pos = np.concatenate([preds.reshape(n, 6), prey], -1)
    prev_pos = np.concatenate([prev_preds.reshape(n, 6), prev_prey], -1)
    vel = pos - prev_pos
    feats = np.zeros((n, N_PREDATORS, PREDATOR_FEATURE_DIM), np.float32)
    for p in range(N_PREDATORS):
        pid = np.zeros((n, N_PREDATORS), np.float32)
        pid[:, p] = 1.0
        feats[:, p] = np.concatenate([pos, vel, pid], -1)
    return feats


def build_predator_samples(
    ds: dict[str, np.ndarray],
    t0: int,
    t1: int,
    ep_len: int = EP_LEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(state, action, episode_id)`` samples over a step interval.

    The returned order is timestep outer, predator inner, episode vector inner.
    Keeping that order stable matters for GIF/debug tooling that maps model
    correctness back onto ``(episode, step, predator)`` grids.
    """

    prey = ds["prey_pos"]
    preds = ds["pred_pos"]
    acts = ds["pred_act"]
    if prey.ndim != 3 or prey.shape[-1] != 2:
        raise ValueError("ds['prey_pos'] must have shape (N, T + 1, 2)")
    if preds.ndim != 4 or preds.shape[2:] != (N_PREDATORS, 2):
        raise ValueError("ds['pred_pos'] must have shape (N, T + 1, 3, 2)")
    if acts.shape[:2] != (len(prey), preds.shape[1] - 1):
        raise ValueError("ds['pred_act'] must have shape (N, T, 3)")
    if t1 > acts.shape[1]:
        raise ValueError(f"t1={t1} exceeds action horizon {acts.shape[1]}")

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    ep_ids = np.arange(len(prey), dtype=np.int32)
    for t in valid_bc_steps(t0, t1, ep_len):
        feats = predator_state_features(
            preds[:, t],
            prey[:, t],
            preds[:, t - 1],
            prey[:, t - 1],
        )
        for p in range(N_PREDATORS):
            states.append(feats[:, p])
            actions.append(acts[:, t, p])
            episodes.append(ep_ids)

    return (
        np.concatenate(states).astype(np.float32),
        np.concatenate(actions).astype(np.int32),
        np.concatenate(episodes).astype(np.int32),
    )
