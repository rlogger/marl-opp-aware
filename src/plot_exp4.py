"""Plot Exp 4 training curves: IQL vs OA-IQL vs MAPPO on the resource env.

Reads the *_metrics.npz files produced by each trainer and generates:
  1. Greedy-policy return curves (mean +/- std across 3 seeds)
     - IQL / OA-IQL: test returns (greedy eval during training)
     - MAPPO: training returns (on-policy, already near-greedy)
  2. Resource collection rate curves
  3. A final-return summary table

The x-axis is environment timesteps (not update steps) so curves
from different algorithms are directly comparable.

Usage:
    python src/plot_exp4.py [--placement random]
    python src/plot_exp4.py --placement circle
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGDIR = "logs/MPE_simple_tag_v3"
ENV    = "MPE_simple_tag_v3"

# Steps-per-update for each algorithm family (NUM_ENVS * NUM_STEPS)
STEPS_PER_UPDATE = {
    "IQL":    8 * 26,     # 208
    "OA-IQL": 8 * 26,     # 208
    "MAPPO":  32 * 128,   # 4096
}


def load_metrics(alg_name, seed=0):
    path = os.path.join(LOGDIR, f"{alg_name}_{ENV}_seed{seed}_metrics.npz")
    if not os.path.exists(path):
        return None
    return dict(np.load(path))


def extract_curves(m, team, label):
    """Extract the most appropriate return curves for each algorithm.

    For IQL / OA-IQL: use test (greedy-eval) returns since training
    returns are epsilon-greedy and not comparable to MAPPO.
    For MAPPO: use training returns (on-policy, near-greedy).
    """
    if label in ("IQL", "OA-IQL"):
        key = f"test__{team}__returned_episode_returns"
    else:
        key = f"{team}__returned_episode_returns"

    if key not in m:
        # Fallback: try training returns
        key = f"{team}__returned_episode_returns"
    if key not in m:
        return None

    arr = m[key]  # (num_seeds, num_updates)
    if arr.ndim > 2:
        arr = arr.mean(axis=tuple(range(2, arr.ndim)))
    return arr  # (num_seeds, T)


def extract_resources(m, team):
    key = f"{team}__resources_collected"
    if key not in m:
        return None
    arr = m[key]
    if arr.ndim > 2:
        arr = arr.mean(axis=tuple(range(2, arr.ndim)))
    return arr


def smooth(x, w=20):
    """Simple moving average."""
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def resample_to_timesteps(arr, steps_per_update, total_ts=2_000_000, n_out=500):
    """Resample curves from update-step domain to a uniform timestep grid."""
    n_seeds, T = arr.shape
    ts_in = np.arange(T) * steps_per_update
    ts_out = np.linspace(0, total_ts, n_out)
    out = np.empty((n_seeds, n_out))
    for s in range(n_seeds):
        out[s] = np.interp(ts_out, ts_in, arr[s])
    return ts_out, out


def plot_team_curves(ax, data_dict, team, title, window=20, total_ts=2_000_000):
    """Plot mean+/-std for each algorithm on a common timestep x-axis."""
    colors = {"IQL": "#2196F3", "OA-IQL": "#FF9800", "MAPPO": "#4CAF50"}

    for label, m in data_dict.items():
        arr = extract_curves(m, team, label)
        if arr is None:
            continue

        # Smooth in update-step domain, then resample to timesteps
        smoothed = np.array([smooth(arr[i], window) for i in range(arr.shape[0])])
        spu = STEPS_PER_UPDATE.get(label, 208)
        ts, resampled = resample_to_timesteps(smoothed, spu, total_ts)

        mean = resampled.mean(axis=0)
        std = resampled.std(axis=0)
        c = colors.get(label, "#999")
        x_millions = ts / 1e6
        ax.plot(x_millions, mean, label=label, color=c, linewidth=1.5)
        ax.fill_between(x_millions, mean - std, mean + std, alpha=0.18, color=c)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Environment timesteps (M)")
    ax.set_ylabel("Episode return")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_resource_curves(ax, data_dict, team, title, window=20, total_ts=2_000_000):
    """Plot resource collection rate."""
    colors = {"IQL": "#2196F3", "OA-IQL": "#FF9800", "MAPPO": "#4CAF50"}

    for label, m in data_dict.items():
        arr = extract_resources(m, team)
        if arr is None:
            continue
        smoothed = np.array([smooth(arr[i], window) for i in range(arr.shape[0])])
        spu = STEPS_PER_UPDATE.get(label, 208)
        ts, resampled = resample_to_timesteps(smoothed, spu, total_ts)
        mean = resampled.mean(axis=0)
        std = resampled.std(axis=0)
        c = colors.get(label, "#999")
        x_millions = ts / 1e6
        ax.plot(x_millions, mean, label=label, color=c, linewidth=1.5)
        ax.fill_between(x_millions, mean - std, mean + std, alpha=0.18, color=c)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Environment timesteps (M)")
    ax.set_ylabel("Resources collected / step")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement", default="random",
                    choices=["circle", "corners", "random"])
    ap.add_argument("--window", type=int, default=20,
                    help="Smoothing window (default 20)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = args.placement

    alg_names = {
        "IQL":    f"iql_teams_resources_{p}",
        "OA-IQL": f"iql_teams_oa_resources_{p}",
        "MAPPO":  f"mappo_teams_resources_{p}",
    }

    data = {}
    for label, alg in alg_names.items():
        m = load_metrics(alg)
        if m is not None:
            data[label] = m
            print(f"Loaded {label}: {alg} ({len(m)} keys)")
        else:
            print(f"Missing {label}: {alg}")

    if not data:
        print("No metrics found. Train first.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    plot_team_curves(axes[0, 0], data, "pred", f"Predator return ({p})",
                     window=args.window)
    plot_team_curves(axes[0, 1], data, "prey", f"Prey return ({p})",
                     window=args.window)
    plot_resource_curves(axes[1, 0], data, "pred", f"Predator resource collection ({p})",
                         window=args.window)
    plot_resource_curves(axes[1, 1], data, "prey", f"Prey resource collection ({p})",
                         window=args.window)

    fig.suptitle(f"Exp 4: Resource collection — {p} placement\n"
                 f"(IQL/OA-IQL: greedy test returns; MAPPO: on-policy returns)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = args.out or f"plots/exp4_training_{p}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    # Print final-return summary table
    print(f"\n{'':>10} {'Pred return':>15} {'Prey return':>15} {'Pred res':>10} {'Prey res':>10}")
    print("-" * 65)
    for label, m in data.items():
        row = f"{label:>10}"
        for team in ["pred", "prey"]:
            arr = extract_curves(m, team, label)
            if arr is not None:
                final = arr[:, -20:].mean()
                std = arr[:, -20:].mean(axis=1).std()
                row += f" {final:>+10.1f}+/-{std:.1f}"
            else:
                row += f" {'N/A':>14}"
        for team in ["pred", "prey"]:
            arr = extract_resources(m, team)
            if arr is not None:
                row += f" {arr[:, -20:].mean():>8.3f}"
            else:
                row += f" {'N/A':>8}"
        print(row)


if __name__ == "__main__":
    main()
