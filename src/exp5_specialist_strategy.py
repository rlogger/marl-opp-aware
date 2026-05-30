"""Specialist prey express distinct, encodable strategies (Deliverables 1 & 2).

Earlier we found that a single prey trained on MIXED placements does not express
placement-distinct behaviour per episode (it is dominated and evades). Here we
train SEPARATE specialists — one prey co-trained only on the circle layout, one
only on corners — and ask the two questions the status slide poses:

  Deliverable 1: do different resource placements -> different strategies?
      Measure: 2D occupancy density of each specialist + a supervised linear
      probe on the raw trajectory (accuracy >> majority class = separable).

  Deliverable 2: can a VAE latent encode the strategy?
      Measure: train the unsupervised TrajVAE on the pooled 2-specialist set,
      cluster the latent (GMM k=2) vs the specialist label (ARI/NMI), and probe
      the latent. Clean separation = the encoder recovers strategy.

Each specialist is rolled out in its NATIVE env (the layout it trained on), so
the placement is fixed across the whole trajectory (no auto-reset mixing).

Usage:
    python src/exp5_specialist_strategy.py --algorithm iql
    python src/exp5_specialist_strategy.py --algorithm mappo
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_trajectory_dataset_resources as G
from generate_trajectory_dataset_resources import rollout_one_checkpoint
from exp4_vae_separation import probe_acc
from exp5b_encoder_length import occupancy

LOGDIR = "logs/MPE_simple_tag_v3"
PLOTDIR = "plots"
PLACEMENTS = ["circle", "corners"]
SEEDS = [0, 1, 2]
NUM_EPS = 100
COL = {"circle": "#0d6e7a", "corners": "#b85c10"}


def collect(algorithm, ckpt_suffix=""):
    """Roll out each specialist in its native env. Returns pooled arrays."""
    pos, lab, res = [], [], []
    for k, placement in enumerate(PLACEMENTS):
        for seed in SEEDS:
            rng = jax.random.PRNGKey(7000 + seed)
            d = rollout_one_checkpoint(algorithm, placement + ckpt_suffix, seed,
                                       NUM_EPS, placement, rng)
            pos.append(d["positions"].astype(np.float32))   # (N, T+1, 2)
            res.append(d["resource_pos"][:, 0].astype(np.float32))  # (N,4,2)
            lab.append(np.full(d["positions"].shape[0], k, np.int32))
            print(f"  [{algorithm}] {placement} seed{seed}  N={d['positions'].shape[0]}",
                  flush=True)
    return (np.concatenate(pos), np.concatenate(lab), np.concatenate(res))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", default="iql", choices=["iql", "oa_iql", "mappo"])
    p.add_argument("--ckpt_suffix", default="",
                   help="e.g. '_wp' for the weakened-predator specialists")
    p.add_argument("--pred_max_speed", type=float, default=None)
    p.add_argument("--pred_accel", type=float, default=None)
    p.add_argument("--collect_reward", type=float, default=5.0)
    p.add_argument("--tag", default="",
                   help="output filename tag (default = algorithm)")
    args = p.parse_args()
    os.makedirs(PLOTDIR, exist_ok=True)

    # match the eval env to how the checkpoints were trained
    G.PRED_MAX_SPEED = args.pred_max_speed
    G.PRED_ACCEL = args.pred_accel
    G.COLLECT_REWARD = args.collect_reward
    tag = args.tag or args.algorithm

    pos, labels, res0 = collect(args.algorithm, args.ckpt_suffix)
    N = len(pos)
    majority = max((labels == 0).mean(), (labels == 1).mean())

    # Deliverable 1: is the strategy in the data? Probe the per-trajectory
    # occupancy histogram (the representation that exposes where the prey goes).
    occ = np.stack([occupancy(t[1:]) for t in pos])      # (N, bins^2)
    occ_std = (occ - occ.mean(0)) / (occ.std(0) + 1e-6)
    occ_probe = probe_acc(occ_std, labels)
    print(f"\n[{args.algorithm}] Deliverable 1 occupancy probe = {occ_probe:.2f} "
          f"(majority {majority:.2f})")

    np.savez(os.path.join(LOGDIR, f"exp5_specialist_{tag}.npz"),
             occ_probe=occ_probe, majority=majority, labels=labels)

    # ---------------------------------------------------------------- figure #
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    for ax, k, placement in zip(axes, [0, 1], PLACEMENTS):
        m = labels == k
        xy = pos[m][:, 1:].reshape(-1, 2)
        hb = ax.hexbin(xy[:, 0], xy[:, 1], gridsize=34, cmap="magma", mincnt=1,
                       extent=(-1.6, 1.6, -1.6, 1.6))
        rr = res0[m].reshape(-1, 2)
        ax.scatter(rr[:, 0], rr[:, 1], s=40, facecolors="none",
                   edgecolors="#39d0ff", linewidths=1.3, label="resources")
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.set_title(f"{placement}-specialist prey", fontsize=13)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(f"Deliverable 1 — {args.algorithm.upper()} prey learns a "
                 f"distinct route per resource layout\n(occupancy linearly "
                 f"separable at {occ_probe:.2f} vs {majority:.2f} chance)",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(PLOTDIR, f"exp5_specialist_{tag}.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
