"""A/B comparison plots: baseline IQL vs opponent-aware IQL.

Inputs:
    --baseline logs/MPE_simple_tag_v3/iql_teams_MPE_simple_tag_v3_seed0_metrics.npz
    --oa       logs/MPE_simple_tag_v3/iql_teams_oa_MPE_simple_tag_v3_seed0_metrics.npz
Outputs:
    plots/compare_test_returns.png
    plots/compare_opp_modeling.png
    plots/compare_summary.png
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p):
    d = np.load(p)
    return {k: d[k] for k in d.files}


def test_points(arr):
    """Test metrics are step-held between evals — return (x, y) at value changes."""
    y = np.asarray(arr)
    if y.ndim == 2:
        y = y[0]
    dy = np.diff(y, prepend=y[0] - 1)
    idx = np.where(np.abs(dy) > 1e-9)[0]
    if len(idx) == 0:
        idx = np.array([0, len(y) - 1])
    return idx, y[idx]


def smooth(y, w=31):
    y = np.asarray(y)
    if y.ndim == 2:
        y = y[0]
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--oa", required=True)
    p.add_argument("-o", "--out", default="plots")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    B = load(args.baseline)
    O = load(args.oa)

    # ---------- 1) A/B greedy test return (pred + prey) ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, team, title in [
        (axes[0], "pred", "Predator greedy test return"),
        (axes[1], "prey", "Prey greedy test return"),
    ]:
        key = f"test__{team}__returned_episode_returns"
        for label, data, color in [
            ("baseline IQL", B, "tab:gray"),
            ("OA-IQL", O, "tab:red" if team == "pred" else "tab:blue"),
        ]:
            if key not in data:
                continue
            idx, vals = test_points(data[key])
            ls = "--" if label == "baseline IQL" else "-"
            ax.plot(idx, vals, color=color, label=label, lw=2, marker="o", ms=3, linestyle=ls)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_xlabel("update step")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Baseline IQL vs Opponent-Aware IQL  |  MPE simple_tag")
    fig.tight_layout()
    fn = os.path.join(args.out, "compare_test_returns.png")
    fig.savefig(fn, dpi=140)
    print(f"wrote {fn}")

    # ---------- 2) Opponent prediction accuracy + CE (only OA has these) ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, suffix, title, ylim in [
        (axes[0], "opp_acc", "Opponent-action prediction accuracy", (0, 1)),
        (axes[1], "opp_ce", "Opponent-action cross-entropy loss", None),
    ]:
        for team, color in [("pred", "tab:red"), ("prey", "tab:blue")]:
            key = f"{team}__{suffix}"
            if key not in O:
                continue
            y = smooth(O[key], 51)
            ax.plot(np.arange(len(y)), y, color=color, label=team, lw=1.8)
        if suffix == "opp_acc":
            ax.axhline(0.2, color="k", lw=0.8, ls=":", alpha=0.6, label="random (1/5)")
        ax.set_xlabel("update step")
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("OA-IQL  |  auxiliary opponent-action head")
    fig.tight_layout()
    fn = os.path.join(args.out, "compare_opp_modeling.png")
    fig.savefig(fn, dpi=140)
    print(f"wrote {fn}")

    # ---------- 3) Summary bar: final greedy return ----------
    def final_test(data, team):
        key = f"test__{team}__returned_episode_returns"
        return float(np.asarray(data[key])[0, -1])

    labels = ["baseline IQL", "OA-IQL"]
    pred_vals = [final_test(B, "pred"), final_test(O, "pred")]
    prey_vals = [final_test(B, "prey"), final_test(O, "prey")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    xs = np.arange(len(labels))
    axes[0].bar(xs, pred_vals, color=["tab:gray", "tab:red"])
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("final greedy test return")
    axes[0].set_title("Predator team (↑ better)")
    for i, v in enumerate(pred_vals):
        axes[0].text(i, v + 0.5, f"{v:+.2f}", ha="center", fontweight="bold")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(xs, prey_vals, color=["tab:gray", "tab:blue"])
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("final greedy test return")
    axes[1].set_title("Prey team (↑ better)")
    for i, v in enumerate(prey_vals):
        axes[1].text(i, v + 1, f"{v:+.2f}", ha="center", fontweight="bold")
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("Final greedy test return (last eval of 2M-step training)")
    fig.tight_layout()
    fn = os.path.join(args.out, "compare_summary.png")
    fig.savefig(fn, dpi=140)
    print(f"wrote {fn}")

    # ---------- CLI summary ----------
    print("\n=== A/B summary ===")
    print(f"{'team':<6}  {'baseline':>12}  {'OA-IQL':>12}  {'Δ':>10}")
    for team, bv, ov in [("pred", pred_vals[0], pred_vals[1]),
                          ("prey", prey_vals[0], prey_vals[1])]:
        print(f"{team:<6}  {bv:>+12.3f}  {ov:>+12.3f}  {ov - bv:>+10.3f}")


if __name__ == "__main__":
    main()
