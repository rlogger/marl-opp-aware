"""6/5 meeting deliverable: does conditioning predator BC on the unsupervised
opponent latent help?  pi(a | s)  vs  pi(a | s, z).

  * naive BC (the notes' starting point): predict a predator's action from all
    agent locations (+ a one-step velocity proxy + predator id).
  * latent-conditioned BC: append z = the unsupervised encoder's latent of the
    PREY's episode-1 OCCUPANCY (the opponent-strategy code: where it routes;
    raw coordinate windows verifiably carry no placement signal). Variants:
    z from the VAE, z from the JEPA encoder (predict episode-1+2 occupancy
    representation from episode 1), and the oracle placement one-hot (ceiling).

Protocol: watch one episode, predict the next -- z is computed from episode 1
(steps 0..25) only, and BC samples are episode 2 (steps 26..50), so the
conditioning never sees the predicted steps. Encoders are pretrained
unsupervised on the pooled unlabeled prey trajectories (standard SSL
pretraining; no label enters any latent). BC uses an EPISODE-level train/val
split. mean +/- std over 3 BC seeds (encoder latent from encoder seed 0).

Outputs: plots/mopa_bc_latent_{alg}.png
         logs/MPE_simple_tag_v3/mopa_bc_latent_{alg}.npz
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mopa import legacy
from mopa.bc import bc_comparison
from mopa.data import specialist_dataset, standardize, occupancy, EP_LEN
from mopa.encoders import train_jepa, train_vae
import jax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", default="mappo", choices=["mappo", "iql"])
    ap.add_argument("--n_eps", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    os.makedirs(legacy.PLOTDIR, exist_ok=True)

    print(f"rolling out specialists ({args.algorithm}, wp)...")
    ds = specialist_dataset(algorithm=args.algorithm, n_eps=args.n_eps)
    y = ds["label"]
    print(f"  episodes: {len(y)}")

    # unsupervised prey-strategy latents from episode-1 occupancy (seed 0)
    Xc, _, _ = standardize(occupancy(ds["prey_pos"], 0, EP_LEN))
    Xt, _, _ = standardize(occupancy(ds["prey_pos"], 0, 2 * EP_LEN))
    print("pretraining unsupervised encoders on prey episode-1 occupancy...")
    z_vae = train_vae(Xc, jax.random.PRNGKey(0))
    z_jepa = train_jepa(Xc, Xt, jax.random.PRNGKey(0))
    oracle = np.eye(2, dtype=np.float32)[y]

    print("behaviour cloning episode 2 (episode-level split, 3 seeds)...")
    res = bc_comparison(
        ds,
        {"none": None, "z_vae": z_vae, "z_jepa": z_jepa, "oracle": oracle},
        ctx=EP_LEN + 1, seeds=tuple(args.seeds))

    np.savez(os.path.join(legacy.LOGDIR, f"mopa_bc_latent_{args.algorithm}.npz"),
             **{f"{k}_runs": v[2] for k, v in res.items()},
             **{f"{k}_mean": v[0] for k, v in res.items()},
             **{f"{k}_std": v[1] for k, v in res.items()},
             ctx=26)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    order = ["none", "z_vae", "z_jepa", "oracle"]
    labels = ["$\\pi(a|s)$\nnaive BC", "$\\pi(a|s,z_{VAE})$",
              "$\\pi(a|s,z_{JEPA})$", "$\\pi(a|s,$placement$)$\noracle"]
    vals = [res[k][0] for k in order]
    errs = [res[k][1] for k in order]
    bars = ax.bar(labels, vals, yerr=errs, capsize=5,
                  color=["#999", "#7f9c9f", "#0d6e7a", "#073b42"])
    base = res["none"][0]
    ax.axhline(base, ls=":", c="k", lw=1)
    ax.set_ylabel("held-out action accuracy (episode split)")
    ax.set_title(f"Predator BC with an opponent-strategy latent "
                 f"({args.algorithm.upper()}, episode-1 occupancy latent)\n"
                 "conditioning on the unsupervised prey latent improves "
                 "action prediction")
    lo = min(vals) - 3 * max(errs) - 0.02
    ax.set_ylim(max(0.0, lo), max(vals) + 3 * max(errs) + 0.02)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004,
                f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(legacy.PLOTDIR, f"mopa_bc_latent_{args.algorithm}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
