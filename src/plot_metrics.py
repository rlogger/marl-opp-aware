"""Plot training curves (predator vs prey) from metrics npz produced by iql_teams.py."""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth(y, w=21):
    if y.ndim == 1:
        y = y[None]
    if y.shape[-1] < w:
        return y.mean(0), y.std(0)
    kernel = np.ones(w) / w
    sm = np.stack([np.convolve(yi, kernel, mode="valid") for yi in y])
    return sm.mean(0), sm.std(0)


def test_eval_points(y):
    """Test metrics are step-held between evals; return just the (x, y) where value changes."""
    y = np.asarray(y)
    if y.ndim == 2:
        y = y[0]
    dy = np.diff(y, prepend=y[0] - 1)
    idx = np.where(np.abs(dy) > 1e-9)[0]
    if len(idx) == 0:
        idx = np.array([0, len(y) - 1])
    return idx, y[idx]


def load(npz_path):
    d = np.load(npz_path)
    return {k: d[k] for k in d.files}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="path to metrics npz")
    p.add_argument("-o", "--out", default="plots")
    p.add_argument("--smooth", type=int, default=21)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    m = load(args.npz)

    # --- Panel 1: smoothed train return (per-step-averaged rollout signal) ---
    # --- Panel 2: greedy test return per evaluation (the actual eval signal) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    for team, color in [("pred", "tab:red"), ("prey", "tab:blue")]:
        key = f"{team}__returned_episode_returns"
        if key not in m:
            continue
        y = np.asarray(m[key])
        if y.ndim == 1:
            y = y[None]
        mu, sd = smooth(y, w=args.smooth)
        x = np.arange(len(mu))
        ax.plot(x, mu, color=color, label=team, lw=1.8)
        ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.2)
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("update step")
    ax.set_title("Train return (rollout, step-averaged)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for team, color in [("pred", "tab:red"), ("prey", "tab:blue")]:
        key = f"test__{team}__returned_episode_returns"
        if key not in m:
            continue
        idx, vals = test_eval_points(m[key])
        ax.plot(idx, vals, color=color, label=team, lw=1.8, marker="o", ms=3)
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("update step")
    ax.set_title("Greedy test return (per 30-step eval episode)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("MPE simple_tag  |  independent two-team IQL (co-training)")
    fig.tight_layout()
    out_path = os.path.join(args.out, "train_curves.png")
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")

    # loss + Q-values
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, key_suffix, title in [
        (axes2[0], "loss", "TD loss"),
        (axes2[1], "qvals", "Mean Q"),
    ]:
        for team, color in [("pred", "tab:red"), ("prey", "tab:blue")]:
            key = f"{team}__{key_suffix}"
            if key not in m:
                continue
            y = np.asarray(m[key])
            if y.ndim == 1:
                y = y[None]
            mu, sd = smooth(y, w=args.smooth)
            x = np.arange(len(mu))
            ax.plot(x, mu, color=color, label=team, lw=1.8)
            ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.2)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_xlabel("update step")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig2.suptitle("MPE simple_tag  |  TD loss and mean Q per team")
    fig2.tight_layout()
    out_path2 = os.path.join(args.out, "loss_q.png")
    fig2.savefig(out_path2, dpi=140)
    print(f"wrote {out_path2}")

    print("\navailable metric keys:")
    for k in sorted(m):
        v = np.asarray(m[k])
        print(f"  {k}  shape={v.shape}")


if __name__ == "__main__":
    main()
