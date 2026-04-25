"""Plots for Experiment 3 (MLP + tree-expanded planning + cross-play tournament):

  1. Training curves (mean +/- std over 3 seeds): IQL-MLP vs OA-IQL-MLP.
     - Predator test return
     - Prey test return
     - OA-head accuracy (pred + prey) for OA variant only
  2. Tournament heatmaps (3x3 matrix of pred_return mean, std, captures/ep) from
     tournament.npz.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGDIR = "logs/MPE_simple_tag_v3"
OUTDIR = "plots"


def test_eval_points(y):
    """Test metrics are step-held between evals; return just the (idx, mean, std) where value changes."""
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[None]
    ref = y[0]
    dy = np.diff(ref, prepend=ref[0] - 1)
    idx = np.where(np.abs(dy) > 1e-9)[0]
    if len(idx) == 0:
        idx = np.array([0, len(ref) - 1])
    sub = y[:, idx]
    return idx, sub.mean(0), sub.std(0)


def smooth(y, w=51):
    if y.ndim == 1:
        y = y[None]
    if y.shape[-1] < w:
        return y.mean(0), y.std(0), np.arange(y.shape[-1])
    kernel = np.ones(w) / w
    sm = np.stack([np.convolve(yi, kernel, mode="valid") for yi in y])
    return sm.mean(0), sm.std(0), np.arange(sm.shape[-1])


def load_metrics(tag):
    path = f"{LOGDIR}/{tag}_MPE_simple_tag_v3_seed0_metrics.npz"
    return np.load(path)


def plot_training_curves():
    mlp = load_metrics("iql_teams_mlp")
    oa = load_metrics("iql_teams_oa_mlp")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: Predator test return
    ax = axes[0]
    for tag, m, color, label in [
        ("MLP", mlp, "tab:gray", "IQL-MLP (baseline)"),
        ("OA-MLP", oa, "tab:green", "OA-IQL-MLP"),
    ]:
        idx, mu, sd = test_eval_points(m["test__pred__returned_episode_returns"])
        es = m["env_step"][0][idx]
        ax.plot(es, mu, color=color, label=label, lw=2.0, marker="o", ms=3)
        ax.fill_between(es, mu - sd, mu + sd, color=color, alpha=0.18)
    ax.set_xlabel("env step")
    ax.set_ylabel("return / episode")
    ax.set_title("Predator test return (3 seeds)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # Panel 2: Prey test return
    ax = axes[1]
    for tag, m, color, label in [
        ("MLP", mlp, "tab:gray", "IQL-MLP (baseline)"),
        ("OA-MLP", oa, "tab:green", "OA-IQL-MLP"),
    ]:
        idx, mu, sd = test_eval_points(m["test__prey__returned_episode_returns"])
        es = m["env_step"][0][idx]
        ax.plot(es, mu, color=color, label=label, lw=2.0, marker="o", ms=3)
        ax.fill_between(es, mu - sd, mu + sd, color=color, alpha=0.18)
    ax.set_xlabel("env step")
    ax.set_ylabel("return / episode")
    ax.set_title("Prey test return (3 seeds)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Panel 3: OA-head accuracy (OA variant only)
    ax = axes[2]
    for team, color, label in [
        ("pred", "tab:red", "pred opp-head acc"),
        ("prey", "tab:blue", "prey opp-head acc"),
    ]:
        key = f"{team}__opp_acc"
        if key not in oa:
            continue
        y = np.asarray(oa[key])
        mu, sd, xi = smooth(y, w=101)
        es = oa["env_step"][0][:len(mu)]
        ax.plot(es, mu, color=color, label=label, lw=2.0)
        ax.fill_between(es, mu - sd, mu + sd, color=color, alpha=0.18)
    ax.axhline(1/5, color="k", lw=0.7, ls="--", alpha=0.5, label="chance (1/5)")
    ax.set_xlabel("env step")
    ax.set_ylabel("accuracy")
    ax.set_title("Opponent-action prediction accuracy")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.0, 1.0)

    fig.suptitle("Exp 3: IQL-MLP vs OA-IQL-MLP — training curves (mean ± std over 3 seeds)")
    fig.tight_layout()
    out = f"{OUTDIR}/exp3_training_curves.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def plot_tournament():
    d = np.load(f"{LOGDIR}/tournament.npz", allow_pickle=True)
    names = [str(n) for n in d["policy_names"]]
    mats = {
        "pred_return": (d["pred_return_mean"], d["pred_return_std"]),
        "prey_return": (d["prey_return_mean"], d["prey_return_std"]),
        "captures_per_ep": (d["captures_per_ep_mean"], d["captures_per_ep_std"]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (metric, (mu, sd)) in zip(axes, mats.items()):
        # Color by pred-advantage: higher pred_return = redder; higher prey_return = bluer.
        cmap = "RdBu_r" if metric == "pred_return" else ("RdBu" if metric == "prey_return" else "RdBu_r")
        vabs = max(abs(mu.min()), abs(mu.max()))
        im = ax.imshow(mu, cmap=cmap, vmin=-vabs, vmax=vabs, aspect="auto")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mu[i,j]:+.1f}\n±{sd[i,j]:.1f}",
                        ha="center", va="center", fontsize=11,
                        color="white" if abs(mu[i,j]) > 0.55 * vabs else "black",
                        fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_yticks(range(3)); ax.set_yticklabels(names)
        ax.set_xlabel("PREY policy")
        ax.set_ylabel("PRED policy")
        ax.set_title(metric)
        fig.colorbar(im, ax=ax, shrink=0.82)

    n_seeds = int(d["num_seeds"][0])
    n_eps = int(d["num_eps"][0])
    fig.suptitle(
        f"Exp 3: 3×3 cross-play tournament  |  {n_seeds} seeds × {n_eps} eps / cell"
    )
    fig.tight_layout()
    out = f"{OUTDIR}/exp3_tournament.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plot_training_curves()
    plot_tournament()


if __name__ == "__main__":
    main()
