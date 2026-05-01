"""Verify multi-modality of prey trajectory datasets.

Reads logs/MPE_simple_tag_v3/trajectory_dataset_{A,B}.npz produced by
src/generate_trajectory_dataset.py and runs:

  Featurization  -> two parallel feature spaces
    1. Hand-crafted summary features (interpretable; 8-d):
         final_x, final_y, mean_speed, time_in_annulus,
         signed_angular_sum_obs0, signed_angular_sum_obs1,
         x_range, y_range
    2. Position-occupancy histogram (16x16 = 256-d, normalised).

  PCA scatter      -> 2D embedding, colored by ground-truth checkpoint label
                      (Reading A) or by the cluster that GMM assigns (Reading B).

  Cluster sweep    -> KMeans + GaussianMixture for k in [1..9]:
                      ARI / NMI / silhouette / BIC. Plotted vs k.

  Confusion matrix -> Reading A only: GMM(k=9) clusters vs ground-truth labels.

  Sample trajectories per cluster (Reading A: per ground-truth label;
                                   Reading B: per discovered cluster).

All output figures use matplotlib's default backend (Agg-safe) and are saved
under plots/.
"""
import os
import sys
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    confusion_matrix,
)


LOGDIR    = "logs/MPE_simple_tag_v3"
PLOTDIR   = "plots"
OBSTACLES = np.array([[0.5, 0.5], [-0.5, -0.5]], dtype=np.float32)
GRID_H    = 16          # occupancy grid resolution
ARENA     = (-1.4, 1.4)
ANNULUS   = (0.25, 0.6)


# --------------------------- featurization ---------------------------------- #

def hand_features(positions: np.ndarray) -> np.ndarray:
    """positions: (N, T+1, 2). Returns (N, F).

    Init-conditioned: features that depend on the absolute position get the
    initial prey position subtracted, so initial-condition variance does not
    swamp the policy signal.

    Features (in order):
        relative_final_x, relative_final_y,
        mean_speed,
        time_in_annulus_any_obstacle,
        x_range, y_range,
        pooled_mean_sign_cross_in_annulus  (the directional signature),
        pooled_angular_displacement_in_annulus,
        per_obstacle: time_in_band, sign_in_band, ang_sum_in_band   (3 x L = 6).
    """
    N = positions.shape[0]
    prev = positions[:, :-1]                       # (N, T, 2)
    nxt  = positions[:, 1:]                        # (N, T, 2)

    speed = np.linalg.norm(nxt - prev, axis=-1)
    mean_speed = speed.mean(axis=1)

    L = OBSTACLES.shape[0]

    in_band_any  = np.zeros((N, prev.shape[1]), dtype=bool)
    pooled_sign  = np.zeros(N, dtype=np.float32)
    pooled_ang   = np.zeros(N, dtype=np.float32)
    pooled_count = np.zeros(N, dtype=np.float32)
    per_obs = []

    for j in range(L):
        vp = prev - OBSTACLES[j][None, None, :]
        vn = nxt  - OBSTACLES[j][None, None, :]
        d_new = np.linalg.norm(vn, axis=-1)
        in_band_j = (d_new >= ANNULUS[0]) & (d_new <= ANNULUS[1])
        in_band_any |= in_band_j

        cross = vp[..., 0] * vn[..., 1] - vp[..., 1] * vn[..., 0]
        dot   = vp[..., 0] * vn[..., 0] + vp[..., 1] * vn[..., 1]
        ang   = np.arctan2(cross, dot)

        n_in_j = in_band_j.sum(axis=1).clip(min=1)
        sign_in_band_j   = (np.sign(cross) * in_band_j).sum(axis=1) / n_in_j
        ang_sum_in_band  = (ang * in_band_j).sum(axis=1)
        time_in_band_j   = in_band_j.mean(axis=1)
        per_obs.append(np.stack([time_in_band_j, sign_in_band_j, ang_sum_in_band], axis=1))

        # pool across obstacles weighted by in-band steps
        pooled_sign  += (np.sign(cross) * in_band_j).sum(axis=1)
        pooled_ang   += ang_sum_in_band
        pooled_count += in_band_j.sum(axis=1)

    pooled_count = pooled_count.clip(min=1)
    pooled_sign /= pooled_count

    time_in_annulus = in_band_any.mean(axis=1)
    init   = positions[:, 0]
    rel_final = positions[:, -1] - init             # init-conditioned
    x_range = positions[:, :, 0].ptp(axis=1)
    y_range = positions[:, :, 1].ptp(axis=1)

    return np.concatenate(
        [rel_final,
         mean_speed[:, None], time_in_annulus[:, None],
         x_range[:, None], y_range[:, None],
         pooled_sign[:, None], pooled_ang[:, None],
         *per_obs],
        axis=1,
    ).astype(np.float32)


def occupancy_features_relative(positions: np.ndarray, grid_h: int = GRID_H) -> np.ndarray:
    """Init-conditioned occupancy: histogram of (position - initial_position).

    Captures the *shape* of the trajectory, not where it started.
    """
    N = positions.shape[0]
    rel = positions - positions[:, :1]              # (N, T+1, 2)
    edges = np.linspace(-2.0, 2.0, grid_h + 1)
    feats = np.zeros((N, grid_h, grid_h), dtype=np.float32)
    for i in range(N):
        H, _, _ = np.histogram2d(
            rel[i, :, 0], rel[i, :, 1], bins=[edges, edges]
        )
        s = H.sum()
        if s > 0:
            H /= s
        feats[i] = H
    return feats.reshape(N, -1)


def occupancy_features(positions: np.ndarray, grid_h: int = GRID_H) -> np.ndarray:
    """positions: (N, T+1, 2). Returns (N, grid_h*grid_h)."""
    N = positions.shape[0]
    edges = np.linspace(ARENA[0], ARENA[1], grid_h + 1)
    feats = np.zeros((N, grid_h, grid_h), dtype=np.float32)
    for i in range(N):
        H, _, _ = np.histogram2d(
            positions[i, :, 0], positions[i, :, 1], bins=[edges, edges]
        )
        s = H.sum()
        if s > 0:
            H /= s
        feats[i] = H
    return feats.reshape(N, -1)


# --------------------------- analysis --------------------------------------- #

def cluster_sweep(
    X: np.ndarray, y_true: np.ndarray | None, ks=range(1, 10),
    cov_type: str = "full",
) -> dict:
    """Run KMeans + GMM at each k. Returns dict of arrays (one entry per k).

    For high-dim X (> ~30 features), pass cov_type='diag' to avoid singular
    covariance matrices on small N.
    """
    rows = {"k": [], "ari": [], "nmi": [], "silh": [], "bic": []}
    for k in ks:
        km = KMeans(n_clusters=max(k, 2), n_init=20, random_state=0).fit(X)
        gm = GaussianMixture(
            n_components=k, n_init=10, random_state=0,
            covariance_type=cov_type, reg_covar=1e-4,
        ).fit(X)
        ari = adjusted_rand_score(y_true, km.labels_) if y_true is not None and k > 1 else np.nan
        nmi = normalized_mutual_info_score(y_true, km.labels_) if y_true is not None and k > 1 else np.nan
        silh = silhouette_score(X, km.labels_) if k > 1 else np.nan
        bic  = gm.bic(X)
        rows["k"].append(k); rows["ari"].append(ari); rows["nmi"].append(nmi)
        rows["silh"].append(silh); rows["bic"].append(bic)
    return {k: np.array(v) for k, v in rows.items()}


# --------------------------- plotting --------------------------------------- #

def plot_pca(
    X: np.ndarray, y: np.ndarray, title: str, out: str, label_lookup: list | None
):
    pca = PCA(n_components=2).fit(X)
    Xp = pca.transform(X)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    classes = sorted(set(y.tolist()))
    cmap = plt.cm.tab10 if len(classes) <= 10 else plt.cm.tab20
    for i, c in enumerate(classes):
        m = y == c
        lbl = label_lookup[c] if label_lookup is not None else f"cluster {c}"
        ax.scatter(
            Xp[m, 0], Xp[m, 1], s=18, alpha=0.65, color=cmap(i % cmap.N),
            edgecolors="none", label=lbl,
        )
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f} %)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f} %)")
    ax.set_title(title)
    ax.legend(fontsize=8, markerscale=1.4, loc="best", frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


def plot_sweep(sweep: dict, has_labels: bool, title: str, out: str):
    fig, axes = plt.subplots(1, 2 + (2 if has_labels else 0),
                             figsize=(4.0 * (2 + (2 if has_labels else 0)), 3.6))
    ks = sweep["k"]
    axes[0].plot(ks, sweep["bic"], "-o", color="tab:blue")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("BIC (lower better)")
    axes[0].set_title("GMM BIC vs k"); axes[0].grid(alpha=0.3)

    axes[1].plot(ks, sweep["silh"], "-o", color="tab:purple")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("silhouette")
    axes[1].set_title("KMeans silhouette vs k"); axes[1].grid(alpha=0.3)

    if has_labels:
        axes[2].plot(ks, sweep["ari"], "-o", color="tab:orange")
        axes[2].set_xlabel("k"); axes[2].set_ylabel("Adjusted Rand Index")
        axes[2].set_title("ARI vs ground truth"); axes[2].grid(alpha=0.3)
        axes[3].plot(ks, sweep["nmi"], "-o", color="tab:green")
        axes[3].set_xlabel("k"); axes[3].set_ylabel("Normalised Mutual Info")
        axes[3].set_title("NMI vs ground truth"); axes[3].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


def plot_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, label_lookup: list, out: str, title: str
):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_yticklabels(label_lookup, fontsize=8)
    ax.set_xticklabels([f"c{j}" for j in range(cm.shape[1])])
    ax.set_xlabel("GMM cluster"); ax.set_ylabel("checkpoint (ground truth)")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


def plot_sample_trajectories(
    positions: np.ndarray, group_id: np.ndarray, group_lookup: list,
    out: str, title: str, k_per_group: int = 8,
):
    classes = sorted(set(group_id.tolist()))
    n = len(classes)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows),
                             squeeze=False)
    rng = np.random.default_rng(0)
    for idx, c in enumerate(classes):
        ax = axes[idx // cols][idx % cols]
        members = np.where(group_id == c)[0]
        sel = rng.choice(members, size=min(k_per_group, len(members)), replace=False)
        for i, n_idx in enumerate(sel):
            xs = positions[n_idx, :, 0]
            ys = positions[n_idx, :, 1]
            ax.plot(xs, ys, "-", lw=0.9, alpha=0.6,
                    color=plt.cm.tab10(idx % 10))
            ax.plot(xs[0], ys[0], "o", color="black", ms=2.5, alpha=0.7)
        for ox, oy in OBSTACLES:
            ax.add_patch(Circle((ox, oy), 0.18, fill=False,
                                color="cyan", lw=1.2))
        ax.set_xlim(*ARENA); ax.set_ylim(*ARENA)
        ax.set_aspect("equal")
        ax.set_title(group_lookup[c] if c < len(group_lookup) else f"c{c}",
                     fontsize=10)
        ax.tick_params(labelsize=7)
    for j in range(len(classes), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


# --------------------------- main ------------------------------------------- #

def standardise(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-6
    return (X - mu) / sd


def analyse_dataset(npz_path: str, tag: str):
    print(f"\n=== Analysing {tag}: {npz_path} ===")
    d = np.load(npz_path, allow_pickle=True)
    positions    = d["positions"]                    # (N, T+1, 2)
    labels_true  = d["labels"]                       # (N,)
    checkpoints  = list(d["checkpoints"])
    has_labels   = bool((labels_true >= 0).any())

    N, T1, _ = positions.shape
    print(f"  N = {N}, T+1 = {T1}, checkpoints = {len(checkpoints)}, has_labels = {has_labels}")

    # Featurise. The 256-dim occupancy histogram is too high-dim for full-covariance
    # GMM on N=900 samples, so we PCA-reduce to ~30-d first.
    # Use init-conditioned occupancy: histogram of (pos - reset_pos).
    H = standardise(hand_features(positions))
    O_full = occupancy_features_relative(positions)
    O_pca = PCA(n_components=30, random_state=0).fit_transform(O_full)
    O = standardise(O_pca)
    print(f"  hand-features (init-cond)  shape={H.shape}    "
          f"occupancy-relative-PCA-30 shape={O.shape}")

    # PCA scatter
    label_lookup = checkpoints if has_labels else None
    color_y      = labels_true if has_labels else np.zeros(N, dtype=np.int32)
    plot_pca(H, color_y,
             f"{tag} | hand features, PCA",
             f"{PLOTDIR}/multimodal_{tag}_pca_hand.png",
             label_lookup if has_labels else ["all"])
    # PCA scatter on occupancy uses the FULL 256-d feature, not the pre-reduced one
    plot_pca(O_full, color_y,
             f"{tag} | occupancy histogram, PCA",
             f"{PLOTDIR}/multimodal_{tag}_pca_occ.png",
             label_lookup if has_labels else ["all"])

    # Cluster sweep on hand features and on occupancy features
    print("  hand-feature sweep:")
    sweep_h = cluster_sweep(H, labels_true if has_labels else None)
    plot_sweep(sweep_h, has_labels,
               f"{tag} | hand-feature clustering (k sweep)",
               f"{PLOTDIR}/multimodal_{tag}_sweep_hand.png")
    print(f"  k    BIC          silh    ARI     NMI")
    for i, k in enumerate(sweep_h["k"]):
        print(f"  {k:>2d}  {sweep_h['bic'][i]:>+10.1f}  {sweep_h['silh'][i]:+0.3f}  "
              f"{sweep_h['ari'][i]:+0.3f}  {sweep_h['nmi'][i]:+0.3f}")

    print("  occupancy-feature sweep:")
    sweep_o = cluster_sweep(O, labels_true if has_labels else None, cov_type="diag")
    plot_sweep(sweep_o, has_labels,
               f"{tag} | occupancy-feature clustering (k sweep)",
               f"{PLOTDIR}/multimodal_{tag}_sweep_occ.png")
    print(f"  k    BIC          silh    ARI     NMI")
    for i, k in enumerate(sweep_o["k"]):
        print(f"  {k:>2d}  {sweep_o['bic'][i]:>+10.1f}  {sweep_o['silh'][i]:+0.3f}  "
              f"{sweep_o['ari'][i]:+0.3f}  {sweep_o['nmi'][i]:+0.3f}")

    # use the better of the two for the rest of the analysis
    if has_labels:
        ari_h = np.nanmax(sweep_h['ari']); ari_o = np.nanmax(sweep_o['ari'])
        if ari_o > ari_h:
            print(f"  -> using occupancy features for downstream analysis (ari {ari_o:.3f} > {ari_h:.3f})")
            X_best = O
        else:
            print(f"  -> using hand features for downstream analysis (ari {ari_h:.3f} >= {ari_o:.3f})")
            X_best = H
    else:
        X_best = H
    sweep = sweep_h if X_best is H else sweep_o

    # Final clustering for confusion matrix / sample-trajectory plots
    if has_labels:
        K_TRUE = len(checkpoints)
        # Also try collapsing seeds within variant -> 3 clusters
        K_VARIANT = len({c.rsplit("_seed", 1)[0] for c in checkpoints})
        cov_for_best = "diag" if X_best.shape[1] > 20 else "full"
        for K in (K_VARIANT, K_TRUE):
            gm = GaussianMixture(n_components=K, n_init=10, random_state=0,
                                 covariance_type=cov_for_best, reg_covar=1e-4).fit(X_best)
            y_pred = gm.predict(X_best)
            plot_confusion(labels_true, y_pred, checkpoints,
                           f"{PLOTDIR}/multimodal_{tag}_confusion_k{K}.png",
                           f"{tag} | GMM(k={K}) clusters vs ground truth")
        plot_sample_trajectories(positions, labels_true, checkpoints,
                                 f"{PLOTDIR}/multimodal_{tag}_samples_truth.png",
                                 f"{tag} | sample trajectories per checkpoint",
                                 k_per_group=8)
    else:
        ks = sweep["k"]; bic = sweep["bic"]
        k_best = int(ks[int(np.argmin(bic))])
        print(f"  best-BIC k = {k_best}")
        cov_for_best = "diag" if X_best.shape[1] > 20 else "full"
        gm = GaussianMixture(n_components=k_best, n_init=10, random_state=0,
                             covariance_type=cov_for_best, reg_covar=1e-4).fit(X_best)
        y_pred = gm.predict(X_best)
        plot_sample_trajectories(positions, y_pred,
                                 [f"cluster {c}" for c in range(k_best)],
                                 f"{PLOTDIR}/multimodal_{tag}_samples_clusters.png",
                                 f"{tag} | sample trajectories per discovered cluster (k={k_best})",
                                 k_per_group=8)


def main():
    os.makedirs(PLOTDIR, exist_ok=True)
    A = f"{LOGDIR}/trajectory_dataset_A.npz"
    B = f"{LOGDIR}/trajectory_dataset_B.npz"
    if os.path.exists(A): analyse_dataset(A, "A")
    else: print(f"missing {A} — run src/generate_trajectory_dataset.py first")
    if os.path.exists(B): analyse_dataset(B, "B")
    else: print(f"missing {B} — run src/generate_trajectory_dataset.py first")


if __name__ == "__main__":
    main()
