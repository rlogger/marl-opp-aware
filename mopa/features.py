"""Pure trajectory feature extraction utilities.

These helpers intentionally depend only on NumPy so the leakage-sensitive
feature protocol can be unit-tested without importing the JAX training stack.
"""

from __future__ import annotations

import numpy as np

EP_LEN = 25
OCC_BINS = 8
OCC_RANGE = (-1.4, 1.4)


def window(pos: np.ndarray, k: int, kmax: int = EP_LEN) -> np.ndarray:
    """Flatten a fixed first-episode window with masked future steps.

    ``pos`` may be ``(N, T + 1, 2)`` for the prey or ``(N, T + 1, A, 2)`` for a
    team. The first ``k`` rows are retained and rows ``[k, kmax)`` are zeroed.
    """

    if not 0 <= k <= kmax:
        raise ValueError(f"k must be in [0, {kmax}], got {k}")
    ep = pos[:, :kmax]
    flat = ep.reshape(len(ep), kmax, -1).copy()
    flat[:, k:] = 0.0
    return flat.reshape(len(ep), -1).astype(np.float32)


def standardize(
    X: np.ndarray,
    mu: np.ndarray | None = None,
    sd: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features with train-set statistics."""

    if mu is None:
        mu = X.mean(0)
    if sd is None:
        sd = X.std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32), mu, sd


def occupancy(
    pos: np.ndarray,
    t0: int,
    t1: int,
    bins: int = OCC_BINS,
    value_range: tuple[float, float] = OCC_RANGE,
) -> np.ndarray:
    """Per-episode normalized 2-D occupancy histogram over steps ``[t0, t1)``.

    ``pos`` must have shape ``(N, T + 1, 2)``. The output is ``(N, bins*bins)``
    and each non-empty row sums to one.
    """

    if pos.ndim != 3 or pos.shape[-1] != 2:
        raise ValueError("pos must have shape (N, T + 1, 2)")
    if not 0 <= t0 < t1 <= pos.shape[1]:
        raise ValueError(f"invalid occupancy interval [{t0}, {t1})")

    out = np.zeros((len(pos), bins * bins), np.float32)
    hist_range = [value_range, value_range]
    for i in range(len(pos)):
        h, _, _ = np.histogram2d(
            pos[i, t0:t1, 0],
            pos[i, t0:t1, 1],
            bins=bins,
            range=hist_range,
        )
        h = h.ravel().astype(np.float32)
        total = h.sum()
        out[i] = h / total if total > 0 else h
    return out
