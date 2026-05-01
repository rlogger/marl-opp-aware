"""Verify multi-modality in the VAE latent space.

Loads the VAE trained by `src/train_traj_vae.py --agent <agent>` and re-runs
the multi-modality analysis from `src/verify_multimodality.py` on the
posterior-mean latent z = q(z|x).μ instead of hand / occupancy features.

Inputs
    logs/MPE_simple_tag_v3/trajectory_dataset_{A,B}.npz
    logs/MPE_simple_tag_v3/traj_vae_<agent>.safetensors
    logs/MPE_simple_tag_v3/traj_vae_<agent>_stats.npz

Outputs (all under plots/, suffix = agent)
    vae_<agent>_A_pca.png                PCA(z) scatter, coloured by ground-truth checkpoint
    vae_<agent>_A_sweep.png              k-sweep BIC / silhouette / ARI / NMI on z
    vae_<agent>_A_confusion_k3.png       GMM(k=3) clusters vs ground-truth
    vae_<agent>_A_confusion_k9.png       GMM(k=9) clusters vs ground-truth
    vae_<agent>_A_samples_truth.png      sample trajectories per ground-truth checkpoint
    vae_<agent>_A_samples_clusters.png   sample trajectories per VAE-discovered cluster
    vae_<agent>_A_latent_traversal.png   decode along the top-2 principal axes of z
    vae_<agent>_B_pca.png                PCA(z) scatter, coloured by discovered cluster
    vae_<agent>_B_sweep.png              k-sweep BIC / silhouette on z (no labels)
    vae_<agent>_B_samples_clusters.png   sample trajectories per discovered cluster
"""
import argparse
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import jax
import jax.numpy as jnp

from jaxmarl.wrappers.baselines import load_params

# Reuse plotting + analysis helpers from the existing verifier.
from verify_multimodality import (
    OBSTACLES, ARENA,
    cluster_sweep, plot_pca, plot_sweep, plot_confusion,
    plot_sample_trajectories, standardise,
)
from train_traj_vae import TrajVAE, load_and_prepare


LOGDIR  = "logs/MPE_simple_tag_v3"
PLOTDIR = "plots"


# --------------------------- model wrapper ---------------------------------- #

def load_vae(agent: str):
    stats = np.load(os.path.join(LOGDIR, f"traj_vae_{agent}_stats.npz"))
    train_mean = stats["train_mean"].astype(np.float32)
    train_std  = stats["train_std"].astype(np.float32)
    T          = int(stats["T"])
    latent_dim = int(stats["latent_dim"])
    hidden     = int(stats["hidden"])
    out_dim    = T * 2

    model = TrajVAE(hidden=hidden, latent=latent_dim, out_dim=out_dim)
    params = load_params(os.path.join(LOGDIR, f"traj_vae_{agent}.safetensors"))
    return model, params, train_mean, train_std, T, latent_dim


def encode_dataset(model, params, x_raw, train_mean, train_std):
    """Project raw flattened trajectories into the posterior-mean latent."""
    x = (x_raw - train_mean) / train_std
    mu, _ = model.apply(params, jnp.asarray(x), method=TrajVAE.encode)
    return np.asarray(mu)


def decode_latents(model, params, Z, train_mean, train_std, T):
    """Decode an array (M, latent) into trajectories (M, T+1, 2) with t=0=(0,0)."""
    out = model.apply(params, jnp.asarray(Z), method=TrajVAE.decode)
    out = np.asarray(out)
    # un-standardise
    out = out * train_std + train_mean
    M = out.shape[0]
    rel = out.reshape(M, T, 2)
    # prepend the trivial t=0 = (0,0) that we dropped at training time
    zero = np.zeros((M, 1, 2), dtype=rel.dtype)
    return np.concatenate([zero, rel], axis=1)


# --------------------------- analyses --------------------------------------- #

def _agent_traj(d, agent):
    """Pull (N, T+1, 2) trajectory for the named agent from an npz."""
    if agent == "prey":
        return d["positions"]
    i = int(agent.split("_", 1)[1])
    return d["pred_positions"][:, :, i, :]


def analyse(tag: str, agent: str, model, params, train_mean, train_std, latent_dim):
    npz = os.path.join(LOGDIR, f"trajectory_dataset_{tag}.npz")
    print(f"\n=== {agent} | {tag}: {npz} ===")
    d = np.load(npz, allow_pickle=True)
    positions   = _agent_traj(d, agent)
    labels_true = d["labels"]
    checkpoints = list(d["checkpoints"])
    has_labels  = bool((labels_true >= 0).any())

    x_raw, _ = load_and_prepare(npz, agent=agent)
    Z_mu = encode_dataset(model, params, x_raw, train_mean, train_std)
    print(f"  N={positions.shape[0]}, latent shape={Z_mu.shape}, "
          f"per-dim std={Z_mu.std(axis=0).round(3).tolist()}")
    Z = standardise(Z_mu)

    # PCA scatter.
    color_y = labels_true if has_labels else np.zeros(Z.shape[0], dtype=np.int32)
    plot_pca(
        Z, color_y,
        f"{agent} | {tag} | VAE latent z (μ), PCA",
        f"{PLOTDIR}/vae_{agent}_{tag}_pca.png",
        checkpoints if has_labels else ["all"],
    )

    # k-sweep.
    sweep = cluster_sweep(
        Z, labels_true if has_labels else None, cov_type="full"
    )
    plot_sweep(
        sweep, has_labels,
        f"{agent} | {tag} | VAE latent clustering (k sweep)",
        f"{PLOTDIR}/vae_{agent}_{tag}_sweep.png",
    )
    print(f"  k    BIC          silh    ARI     NMI")
    for i, k in enumerate(sweep["k"]):
        print(f"  {int(k):>2d}  {sweep['bic'][i]:>+10.1f}  "
              f"{sweep['silh'][i]:+0.3f}  {sweep['ari'][i]:+0.3f}  "
              f"{sweep['nmi'][i]:+0.3f}")

    # Final clustering for plots.
    if has_labels:
        K_TRUE    = len(checkpoints)
        K_VARIANT = len({c.rsplit("_seed", 1)[0] for c in checkpoints})
        for K in (K_VARIANT, K_TRUE):
            gm = GaussianMixture(
                n_components=K, n_init=10, random_state=0,
                covariance_type="full", reg_covar=1e-4,
            ).fit(Z)
            y_pred = gm.predict(Z)
            plot_confusion(
                labels_true, y_pred, checkpoints,
                f"{PLOTDIR}/vae_{agent}_{tag}_confusion_k{K}.png",
                f"{agent} | {tag} | VAE-latent GMM(k={K}) vs ground truth",
            )

        plot_sample_trajectories(
            positions, labels_true, checkpoints,
            f"{PLOTDIR}/vae_{agent}_{tag}_samples_truth.png",
            f"{agent} | {tag} | sample trajectories per checkpoint (ground truth)",
            k_per_group=8,
        )
        # Also show clusters discovered by GMM at K_TRUE.
        gm = GaussianMixture(
            n_components=K_TRUE, n_init=10, random_state=0,
            covariance_type="full", reg_covar=1e-4,
        ).fit(Z)
        y_pred = gm.predict(Z)
        plot_sample_trajectories(
            positions, y_pred, [f"cluster {c}" for c in range(K_TRUE)],
            f"{PLOTDIR}/vae_{agent}_{tag}_samples_clusters.png",
            f"{agent} | {tag} | sample trajectories per VAE-latent cluster (k={K_TRUE})",
            k_per_group=8,
        )
    else:
        ks = sweep["k"]; bic = sweep["bic"]
        k_best = int(ks[int(np.argmin(bic))])
        print(f"  best-BIC k = {k_best}")
        gm = GaussianMixture(
            n_components=k_best, n_init=10, random_state=0,
            covariance_type="full", reg_covar=1e-4,
        ).fit(Z)
        y_pred = gm.predict(Z)
        plot_sample_trajectories(
            positions, y_pred, [f"cluster {c}" for c in range(k_best)],
            f"{PLOTDIR}/vae_{agent}_{tag}_samples_clusters.png",
            f"{agent} | {tag} | sample trajectories per VAE-latent cluster (k={k_best})",
            k_per_group=8,
        )


def latent_traversal(agent, model, params, train_mean, train_std, T, latent_dim):
    """Decode trajectories along the top-2 principal axes of the encoded
    Dataset A. Each panel is a decoded trajectory. Shows what the latent encodes.
    """
    print(f"\n=== {agent} | latent traversal ===")
    npz = os.path.join(LOGDIR, "trajectory_dataset_A.npz")
    x_raw, _ = load_and_prepare(npz, agent=agent)
    Z_mu = encode_dataset(model, params, x_raw, train_mean, train_std)

    pca = PCA(n_components=2, random_state=0).fit(Z_mu)
    pcs = pca.components_                              # (2, latent)
    z_mean = Z_mu.mean(axis=0)
    z_std  = Z_mu.std(axis=0)
    span = 2.5

    # 5 x 5 grid: rows traverse PC2, cols traverse PC1.
    grid = 5
    a = np.linspace(-span, span, grid)
    b = np.linspace(span, -span, grid)
    Z = []
    for bi in b:
        for ai in a:
            z = z_mean + ai * pcs[0] * z_std.mean() + bi * pcs[1] * z_std.mean()
            Z.append(z)
    Z = np.stack(Z, axis=0)
    trajs = decode_latents(model, params, Z, train_mean, train_std, T)

    fig, axes = plt.subplots(grid, grid, figsize=(2.0 * grid, 2.0 * grid),
                             squeeze=False)
    for i in range(grid):
        for j in range(grid):
            ax = axes[i][j]
            xs = trajs[i * grid + j, :, 0]
            ys = trajs[i * grid + j, :, 1]
            ax.plot(xs, ys, "-", lw=0.9, color="tab:blue")
            ax.plot(xs[0], ys[0], "o", color="black", ms=3.0)
            for ox, oy in OBSTACLES:
                ax.add_patch(Circle((ox, oy), 0.18, fill=False,
                                    color="cyan", lw=1.0))
            ax.set_xlim(*ARENA); ax.set_ylim(*ARENA)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if i == grid - 1:
                ax.set_xlabel(f"PC1 = {a[j]:+.1f}σ", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"PC2 = {b[i]:+.1f}σ", fontsize=8)
    fig.suptitle(f"{agent} | decoded trajectories across the top-2 principal axes of z")
    fig.tight_layout()
    out = f"{PLOTDIR}/vae_{agent}_A_latent_traversal.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="prey",
                   choices=["prey", "pred_0", "pred_1", "pred_2"])
    args = p.parse_args()
    agent = args.agent

    os.makedirs(PLOTDIR, exist_ok=True)
    model, params, mean_, std_, T, latent_dim = load_vae(agent)
    print(f"loaded VAE for {agent}: T={T} latent={latent_dim}")
    analyse("A", agent, model, params, mean_, std_, latent_dim)
    analyse("B", agent, model, params, mean_, std_, latent_dim)
    latent_traversal(agent, model, params, mean_, std_, T, latent_dim)


if __name__ == "__main__":
    main()
