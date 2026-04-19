"""Multi-seed, multi-run comparison plots for the static-obstacles experiments.

Each metrics file is expected to have arrays of shape (NUM_SEEDS, num_updates).
Curves show mean across seeds with a ±std band.

Inputs: metrics .npz from a set of runs.
Outputs (written under --out, default `plots/`):
  static_compare_returns.png    pred + prey greedy-return curves
  static_compare_summary.png    final greedy return bar chart with error bars
  static_compare_loss.png       TD loss + mean |Q| sanity

Any run missing from the CLI is simply skipped. `--random_baseline` is the
original random-obstacles reference (single seed); it is plotted without a
band when only one seed is available.
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p):
    d = np.load(p)
    return {k: d[k] for k in d.files}


def _as_2d(y):
    y = np.asarray(y)
    return y[None, :] if y.ndim == 1 else y  # (seeds, T)


def test_seed_curves(arr):
    """Per-seed greedy-test curves. Returns (idx, values(seeds, len(idx)))."""
    y = _as_2d(arr)                 # (seeds, T)
    seed0 = y[0]
    dy = np.diff(seed0, prepend=seed0[0] - 1)
    idx = np.where(np.abs(dy) > 1e-9)[0]
    if len(idx) == 0:
        idx = np.array([0, len(seed0) - 1])
    return idx, y[:, idx]


def smooth_per_seed(y, w=51):
    y = _as_2d(y)
    if y.shape[1] < w:
        return y
    k = np.ones(w) / w
    return np.stack([np.convolve(s, k, mode="valid") for s in y], axis=0)


RUN_STYLE = {
    "random_baseline": dict(color="#888888", ls="--", lw=1.6, label="random obstacles, no shaping"),
    "static_baseline": dict(color="#1f77b4", ls="-",  lw=1.8, label="static obstacles, no shaping"),
    "static_ccw":      dict(color="#2ca02c", ls="-",  lw=1.8, label="static obstacles, prey CCW bonus"),
    "static_cw":       dict(color="#d62728", ls="-",  lw=1.8, label="static obstacles, prey CW bonus"),
}


def _plot_band(ax, xs, ys, style):
    """Plot mean line + ±std band. ys is (seeds, len(xs))."""
    n = ys.shape[0]
    mean = ys.mean(axis=0)
    ax.plot(xs, mean, marker="o" if n == 1 else None, ms=3, **style)
    if n > 1:
        std = ys.std(axis=0)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.18, color=style["color"], linewidth=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--random_baseline")
    p.add_argument("--static_baseline")
    p.add_argument("--static_ccw")
    p.add_argument("--static_cw")
    p.add_argument("-o", "--out", default="plots")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = {
        "random_baseline": args.random_baseline,
        "static_baseline": args.static_baseline,
        "static_ccw":      args.static_ccw,
        "static_cw":       args.static_cw,
    }
    runs = {name: load(p) for name, p in paths.items() if p}

    # ---------- 1) Greedy test return (pred + prey), mean ± std ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, team, title in [
        (axes[0], "pred", "Predator greedy test return"),
        (axes[1], "prey", "Prey greedy test return"),
    ]:
        key = f"test__{team}__returned_episode_returns"
        for name, data in runs.items():
            if key not in data:
                continue
            idx, per_seed = test_seed_curves(data[key])
            _plot_band(ax, idx, per_seed, RUN_STYLE[name])
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_xlabel("update step")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle("Static obstacles + CW/CCW prey shaping  |  mean \u00b1 std over seeds")
    fig.tight_layout()
    fn = os.path.join(args.out, "static_compare_returns.png")
    fig.savefig(fn, dpi=140); plt.close(fig)
    print(f"wrote {fn}")

    # ---------- 2) Final-return bar chart with error bars ----------
    def final_stats(data, team):
        key = f"test__{team}__returned_episode_returns"
        y = _as_2d(data[key])[:, -1]     # (seeds,)
        return float(y.mean()), float(y.std()) if y.size > 1 else 0.0

    names = list(runs.keys())
    pred = [final_stats(runs[n], "pred") for n in names]
    prey = [final_stats(runs[n], "prey") for n in names]
    pred_mean = [p[0] for p in pred]; pred_std = [p[1] for p in pred]
    prey_mean = [p[0] for p in prey]; prey_std = [p[1] for p in prey]
    colors = [RUN_STYLE[n]["color"] for n in names]
    short = [n.replace("_", "\n") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    xs = np.arange(len(names))
    axes[0].bar(xs, pred_mean, yerr=pred_std, color=colors, capsize=6, error_kw=dict(lw=1.5))
    axes[0].set_xticks(xs); axes[0].set_xticklabels(short, fontsize=8)
    axes[0].set_ylabel("final greedy test return")
    axes[0].set_title("Predator (\u2191 better)")
    for i, (m, s) in enumerate(zip(pred_mean, pred_std)):
        axes[0].text(i, m + (s + 1 if m >= 0 else -s - 1), f"{m:+.1f}\u00b1{s:.1f}", ha="center",
                     fontweight="bold", fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(xs, prey_mean, yerr=prey_std, color=colors, capsize=6, error_kw=dict(lw=1.5))
    axes[1].set_xticks(xs); axes[1].set_xticklabels(short, fontsize=8)
    axes[1].set_ylabel("final greedy test return")
    axes[1].set_title("Prey (\u2191 better)")
    for i, (m, s) in enumerate(zip(prey_mean, prey_std)):
        axes[1].text(i, m + (s + 1 if m >= 0 else -s - 1), f"{m:+.1f}\u00b1{s:.1f}", ha="center",
                     fontweight="bold", fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Final greedy test return (2M steps, mean \u00b1 std)")
    fig.tight_layout()
    fn = os.path.join(args.out, "static_compare_summary.png")
    fig.savefig(fn, dpi=140); plt.close(fig)
    print(f"wrote {fn}")

    # ---------- 3) TD loss + mean |Q| sanity ----------
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for col, team in enumerate(["pred", "prey"]):
        for row, suffix, title in [(0, "loss", f"{team} TD loss"),
                                   (1, "qvals", f"{team} mean |Q|")]:
            ax = axes[row, col]
            key = f"{team}__{suffix}"
            for name, data in runs.items():
                if key not in data:
                    continue
                per_seed = smooth_per_seed(data[key], 101)
                xs = np.arange(per_seed.shape[1])
                _plot_band(ax, xs, per_seed, RUN_STYLE[name])
            ax.set_title(title); ax.grid(alpha=0.3)
            if row == 1: ax.set_xlabel("update step")
            if col == 0 and row == 0:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle("Sanity: TD loss + mean Q (mean \u00b1 std)")
    fig.tight_layout()
    fn = os.path.join(args.out, "static_compare_loss.png")
    fig.savefig(fn, dpi=140); plt.close(fig)
    print(f"wrote {fn}")

    # ---------- CLI summary ----------
    print("\n=== Static experiments summary (mean \u00b1 std over seeds) ===")
    print(f"{'run':<20}  {'n':>3}  {'pred':>15}  {'prey':>15}")
    for n in names:
        npar = _as_2d(runs[n][f"test__pred__returned_episode_returns"]).shape[0]
        pm, ps = final_stats(runs[n], "pred")
        prm, prs = final_stats(runs[n], "prey")
        print(f"{n:<20}  {npar:>3}  {pm:>+8.2f} \u00b1 {ps:>4.2f}  {prm:>+8.2f} \u00b1 {prs:>4.2f}")


if __name__ == "__main__":
    main()
