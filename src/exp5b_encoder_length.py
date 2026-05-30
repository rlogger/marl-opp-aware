"""Deliverable 2, done right: how much opponent data does the encoder need?

The specialist density maps differ (Deliverable 1), but a single 50-step prey
trajectory is dominated by evasion and does not separate. The layout signal is
in WHERE the prey spends time (occupancy), and it accumulates with observation
length. We therefore (a) represent each trajectory by its 2D occupancy
histogram — the featurisation that exposes the route — and (b) sweep the
observation length L, training the VAE encoder at each L and measuring how well
its latent recovers the specialist label.

Result is a separability-vs-length curve: with enough opponent observation the
unsupervised VAE latent cleanly separates circle- from corners-specialist prey.

Usage:
    python src/exp5b_encoder_length.py --algorithm iql
    python src/exp5b_encoder_length.py --algorithm mappo
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_trajectory_dataset_resources as G
from generate_trajectory_dataset_resources import rollout_one_checkpoint
from exp4_vae_separation import probe_acc
from train_traj_vae import TrajVAE, elbo

LOGDIR = "logs/MPE_simple_tag_v3"
PLOTDIR = "plots"
PLACEMENTS = ["circle", "corners"]
SEEDS = [0, 1, 2]
NUM_EPS = 80
L_MAX = 100                      # roll out 100 steps (4 same-layout episodes)
LENGTHS = [25, 50, 75, 100]
BINS = 12                        # occupancy histogram resolution
GRID = (-1.6, 1.6)
COL = {"circle": "#0d6e7a", "corners": "#b85c10"}


def occupancy(traj_xy, bins=BINS):
    """(L,2) trajectory -> normalised (bins*bins,) occupancy histogram."""
    h, _, _ = np.histogram2d(traj_xy[:, 0], traj_xy[:, 1], bins=bins,
                             range=[GRID, GRID])
    h = h.ravel().astype(np.float32)
    s = h.sum()
    return h / s if s > 0 else h


def train_vae_get_z(x_std, latent=3, hidden=64, steps=6000, kl_anneal=1500, seed=0):
    D = x_std.shape[1]
    rng = jax.random.PRNGKey(seed)
    model = TrajVAE(hidden=hidden, latent=latent, out_dim=D)
    rng, ik, ck = jax.random.split(rng, 3)
    params = model.init(ik, jnp.asarray(x_std[:1]), ck)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def step(state, batch, rng, beta):
        def loss_fn(p):
            xh, mu, lv, _ = state.apply_fn(p, batch, rng)
            loss, aux = elbo(xh, batch, mu, lv, beta)
            return loss, aux
        (loss, _), g = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=g), loss

    xt = jnp.asarray(x_std); n = xt.shape[0]
    for s in range(steps):
        rng, bk, sk = jax.random.split(rng, 3)
        idx = jax.random.choice(bk, n, (64,), replace=False)
        beta = float(min(1.0, s / max(kl_anneal, 1)))
        state, _ = step(state, xt[idx], sk, beta)
    mu, _ = model.apply(state.params, xt, method=model.encode)
    return np.asarray(mu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", default="iql", choices=["iql", "oa_iql", "mappo"])
    ap.add_argument("--ckpt_suffix", default="")
    ap.add_argument("--pred_max_speed", type=float, default=None)
    ap.add_argument("--pred_accel", type=float, default=None)
    ap.add_argument("--collect_reward", type=float, default=5.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(PLOTDIR, exist_ok=True)
    tag = args.tag or args.algorithm

    # roll out specialists for L_MAX steps in their native (fixed) env
    G.NUM_STEPS = L_MAX
    G.PRED_MAX_SPEED = args.pred_max_speed
    G.PRED_ACCEL = args.pred_accel
    G.COLLECT_REWARD = args.collect_reward
    pos, labels = [], []
    for k, placement in enumerate(PLACEMENTS):
        for seed in SEEDS:
            rng = jax.random.PRNGKey(9000 + seed)
            d = rollout_one_checkpoint(args.algorithm, placement + args.ckpt_suffix,
                                       seed, NUM_EPS, placement, rng)
            pos.append(d["positions"].astype(np.float32))   # (N, L_MAX+1, 2)
            labels.append(np.full(d["positions"].shape[0], k, np.int32))
            print(f"  [{args.algorithm}] {placement} seed{seed} "
                  f"shape={d['positions'].shape}", flush=True)
    pos = np.concatenate(pos); labels = np.concatenate(labels)
    majority = max((labels == 0).mean(), (labels == 1).mean())

    rows = []
    best = None
    for L in LENGTHS:
        seg = pos[:, 1:L + 1]                          # (N, L, 2)
        occ = np.stack([occupancy(t) for t in seg])    # (N, bins^2)
        occ_std = (occ - occ.mean(0)) / (occ.std(0) + 1e-6)
        # supervised probe on occupancy (is the strategy in the data at length L?)
        sup = probe_acc(occ_std, labels)
        # unsupervised VAE latent on occupancy -> cluster vs label.
        # Standardise z so GMM is not dominated by one high-variance nuisance dim.
        z = train_vae_get_z(occ_std)
        zc = (z - z.mean(0)) / (z.std(0) + 1e-6)
        gm = GaussianMixture(2, covariance_type="full", n_init=20,
                             random_state=0).fit(zc)
        ari = adjusted_rand_score(labels, gm.predict(zc))
        nmi = normalized_mutual_info_score(labels, gm.predict(zc))
        latp = probe_acc(z, labels)
        rows.append((L, sup, latp, ari, nmi))
        print(f"  L={L:3d}  occ-probe={sup:.2f}  latent-probe={latp:.2f}  "
              f"ARI={ari:.2f}  NMI={nmi:.2f}", flush=True)
        if L == LENGTHS[-1]:
            best = (z, PCA(2).fit_transform(z))

    rows = np.array(rows)
    np.savez(os.path.join(LOGDIR, f"exp5b_encoder_length_{tag}.npz"),
             rows=rows, majority=majority, labels=labels)

    # ---------------- figure ---------------- #
    fig, (axc, axs) = plt.subplots(1, 2, figsize=(12, 4.6))

    axc.plot(rows[:, 0], rows[:, 1], "-o", color="#0d6e7a", label="supervised (occupancy)")
    axc.plot(rows[:, 0], rows[:, 2], "-o", color="#b85c10", label="VAE latent probe")
    axc.plot(rows[:, 0], rows[:, 3], "-s", color="#1f7a4e", label="VAE latent ARI (unsup.)")
    axc.axhline(majority, ls="--", c="k", lw=1, label=f"chance ({majority:.2f})")
    axc.set_xlabel("opponent observation length L (steps)")
    axc.set_ylabel("placement recovery")
    axc.set_ylim(-0.05, 1.05)
    axc.set_title("Strategy becomes encodable with enough opponent data")
    axc.legend(fontsize=8); axc.grid(alpha=0.3)

    z, z2 = best
    for k, placement in enumerate(PLACEMENTS):
        m = labels == k
        axs.scatter(z2[m, 0], z2[m, 1], s=14, alpha=0.65, c=COL[placement],
                    label=f"{placement}-prey")
    Lbest, _, latp, ari, _ = rows[-1]
    axs.set_title(f"VAE latent at L={int(Lbest)}  (ARI={ari:.2f}, probe={latp:.2f})")
    axs.set_xlabel("PC1"); axs.set_ylabel("PC2")
    axs.legend(); axs.grid(alpha=0.3)

    fig.suptitle(f"Deliverable 2 — {args.algorithm.upper()} VAE latent recovers "
                 f"prey strategy from occupancy over enough steps", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(PLOTDIR, f"exp5b_encoder_length_{tag}.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
