"""6/5 meeting deliverable: VAE vs JEPA latent encoders on the circle-vs-corners
resource placements -- the strategy axis that is "more complex than just trying
to go to a single part of the map".

Design notes (and how this differs from the failed exp4/exp5 attempt):
  * ABSOLUTE prey coordinates. The placement signal is where the prey routes
    (circle ring r=0.6 vs corner dashes r~1.13); exp4/exp5 init-conditioned the
    trajectories, which removed it (their raw probe sat at chance ~0.5).
  * Specialists rolled out in their native envs with the weakened predators
    they were trained against (PRED_MAX_SPEED=0.6, PRED_ACCEL=1.5), so the prey
    actually expresses its route.
  * Identical protocol to the validated hidden-intent study: context window =
    first CTX steps of the first episode (zero-masked future, fixed window),
    2-D latent, probe + GMM ARI; JEPA additionally predicts the EMA
    representation of the FULL first episode.
  * mean +/- std over 3 encoder seeds; the supervised ceiling is a logistic
    probe on the SAME standardized raw context window (apples to apples).
  * a context-length sweep (the anytime question: who reads the strategy from
    fewer steps?).

Outputs: plots/mopa_latent_resources_{alg}_{team}.png
         logs/MPE_simple_tag_v3/mopa_latent_resources_{alg}_{team}.npz
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mopa.data import (specialist_dataset, window, standardize, occupancy,
                       EP_LEN)
from mopa.encoders import evaluate_encoders, probe_acc
from mopa.paths import log_path, plot_path

COL = {0: "#0d6e7a", 1: "#b85c10"}          # circle teal, corners orange
LBL = {0: "circle", 1: "corners"}


def featurize(ds, team, k, features="window"):
    """features='window': raw absolute coords, first k steps masked-future.
    features='occupancy': context = occupancy over [0, k/2), JEPA target =
    occupancy over [0, k) -- predict WHERE the opponent will spend its time.
    A raw coordinate window does not separate the placements (supervised probe
    at chance, verified); occupancy over longer observation does (exp5b)."""
    pos = ds["prey_pos"] if team == "prey" else ds["pred_pos"]
    if features == "window":
        Xc = window(pos, k)                   # first-k steps, future masked
        Xt = window(pos, EP_LEN)              # full first episode (JEPA target)
    else:
        if team == "pred":
            pos = pos.reshape(pos.shape[0], pos.shape[1], -1, 2)
            Xc = np.concatenate([occupancy(pos[:, :, i], 0, max(2, k // 2))
                                 for i in range(pos.shape[2])], -1)
            Xt = np.concatenate([occupancy(pos[:, :, i], 0, k)
                                 for i in range(pos.shape[2])], -1)
        else:
            Xc = occupancy(pos, 0, max(2, k // 2))
            Xt = occupancy(pos, 0, k)
    Xc, mu, sd = standardize(Xc)
    Xt, _, _ = standardize(Xt)
    return Xc, Xt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", default="mappo", choices=["mappo", "iql"])
    ap.add_argument("--team", default="prey", choices=["prey", "pred"])
    ap.add_argument("--features", default="window",
                    choices=["window", "occupancy"])
    ap.add_argument("--num_steps", type=int, default=None,
                    help="rollout length (default 50; use 100 for occupancy)")
    ap.add_argument("--ctx", type=int, default=12)
    ap.add_argument("--sweep", type=int, nargs="*", default=[4, 8, 12, 18, 25])
    ap.add_argument("--n_eps", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--lat", type=int, default=2)
    ap.add_argument("--enc_steps", type=int, default=5000)
    args = ap.parse_args()

    tag = f"{args.algorithm}_{args.team}"
    if args.features == "occupancy":
        tag += "_occ"
    if args.lat != 2:
        tag += f"_lat{args.lat}"

    print(f"rolling out specialists ({args.algorithm}, wp)...")
    ds = specialist_dataset(algorithm=args.algorithm, n_eps=args.n_eps,
                            num_steps=args.num_steps)
    y = ds["label"]
    print(f"  episodes: {len(y)}  (circle {int((y == 0).sum())}, "
          f"corners {int((y == 1).sum())})")

    # ---- main comparison at ctx ----
    Xc, Xt = featurize(ds, args.team, args.ctx, args.features)
    sup = probe_acc(Xc, y)                     # supervised ceiling, same input
    print(f"  raw-window supervised probe @k={args.ctx}: {sup:.3f} (chance 0.5)")
    summ, z_keep = evaluate_encoders(Xc, Xt, y, n_classes=2,
                                     seeds=tuple(args.seeds), lat=args.lat,
                                     steps=args.enc_steps)
    for n in ("vae", "jepa"):
        print(f"  {n.upper():4s}: probe {summ[f'{n}_probe_mean']:.3f}"
              f" +/- {summ[f'{n}_probe_std']:.3f}   "
              f"ARI {summ[f'{n}_ari_mean']:.3f} +/- {summ[f'{n}_ari_std']:.3f}")

    # ---- context-length sweep ----
    sweep = {"ks": np.asarray(args.sweep, np.int32), "sup": [],
             "vae_probe": [], "vae_std": [], "jepa_probe": [], "jepa_std": []}
    for k in args.sweep:
        Xck, Xtk = featurize(ds, args.team, k, args.features)
        sweep["sup"].append(probe_acc(Xck, y))
        sk, _ = evaluate_encoders(Xck, Xtk, y, n_classes=2,
                                  seeds=tuple(args.seeds), lat=args.lat,
                                  steps=args.enc_steps)
        sweep["vae_probe"].append(sk["vae_probe_mean"])
        sweep["vae_std"].append(sk["vae_probe_std"])
        sweep["jepa_probe"].append(sk["jepa_probe_mean"])
        sweep["jepa_std"].append(sk["jepa_probe_std"])
        print(f"  sweep k={k:2d}: sup {sweep['sup'][-1]:.3f}  "
              f"vae {sweep['vae_probe'][-1]:.3f}  "
              f"jepa {sweep['jepa_probe'][-1]:.3f}")

    np.savez(log_path(f"mopa_latent_resources_{tag}.npz"),
             sup=sup, label=y, ctx=args.ctx,
             vae_z=z_keep["vae"], jepa_z=z_keep["jepa"],
             **{k: v for k, v in summ.items()},
             **{f"sweep_{k}": np.asarray(v) for k, v in sweep.items()})

    # ---- figure: two scatters + bars + sweep ----
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    from sklearn.decomposition import PCA
    for a, name in zip(ax[:2], ("vae", "jepa")):
        z = z_keep[name]
        # AXES: for a 2-D latent the two axes ARE the latent coordinates
        # directly; for a higher-D latent we project to its top-2 principal
        # components, so the axes are PC1/PC2 of the latent (a rotation that
        # keeps the most-spread directions). Each point = one episode's prey
        # trajectory, colored by its true placement.
        if z.shape[1] > 2:
            z = PCA(2).fit_transform(z)
            xl, yl = f"PC1 of {args.lat}-D latent", f"PC2 of {args.lat}-D latent"
        else:
            xl, yl = "latent dim 1", "latent dim 2"
        for lab in (0, 1):
            m = y == lab
            a.scatter(z[m, 0], z[m, 1], s=10, alpha=0.6, c=COL[lab],
                      label=LBL[lab])
        a.set_title(f"{name.upper()} latent (k={args.ctx})\n"
                    f"probe {summ[f'{name}_probe_mean']:.2f}"
                    f"$\\pm${summ[f'{name}_probe_std']:.2f}   "
                    f"ARI {summ[f'{name}_ari_mean']:.2f}"
                    f"$\\pm${summ[f'{name}_ari_std']:.2f}", fontsize=10.5)
        a.set_xlabel(xl)
        a.set_ylabel(yl)
        a.legend(fontsize=8)

    b = ax[2]
    names = ["VAE\nprobe", "JEPA\nprobe", "VAE\nARI", "JEPA\nARI"]
    vals = [summ["vae_probe_mean"], summ["jepa_probe_mean"],
            summ["vae_ari_mean"], summ["jepa_ari_mean"]]
    errs = [summ["vae_probe_std"], summ["jepa_probe_std"],
            summ["vae_ari_std"], summ["jepa_ari_std"]]
    bars = b.bar(names, vals, yerr=errs, capsize=4,
                 color=["#999", "#0d6e7a", "#999", "#0d6e7a"])
    b.axhline(sup, ls="--", c="#1f7a4e", lw=1.4,
              label=f"supervised ceiling {sup:.2f}")
    b.axhline(0.5, ls=":", c="k", lw=1, label="chance (probe) 0.5")
    b.set_ylim(0, 1.05)
    b.set_title("placement recovery (3 encoder seeds)")
    b.legend(fontsize=8)
    for bar, v in zip(bars, vals):
        b.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
               ha="center", fontweight="bold", fontsize=9)

    s = ax[3]
    ks = sweep["ks"]
    s.plot(ks, sweep["sup"], "--", c="#1f7a4e", label="supervised")
    s.errorbar(ks, sweep["vae_probe"], yerr=sweep["vae_std"], marker="o",
               c="#999", label="VAE", capsize=3)
    s.errorbar(ks, sweep["jepa_probe"], yerr=sweep["jepa_std"], marker="o",
               c="#0d6e7a", label="JEPA", capsize=3)
    s.axhline(0.5, ls=":", c="k", lw=1)
    s.set_xlabel("observed steps k")
    s.set_ylabel("probe accuracy")
    s.set_title("how fast is the strategy readable?")
    s.grid(alpha=0.3)
    s.legend(fontsize=8)

    fig.suptitle(f"Circle vs corners ({args.algorithm.upper()} {args.team}): "
                 "unsupervised strategy latents, VAE vs JEPA",
                 fontweight="bold")
    fig.tight_layout()
    out = plot_path(f"mopa_latent_resources_{tag}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
