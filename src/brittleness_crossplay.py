"""OOD-opponent brittleness: 2x2 cross-play of specialist policies.

Deliverable 3 of the status slide. We co-train predators+prey on a fixed
circle layout and (separately) on a fixed corners layout, giving two predator
policies and two prey policies. We then cross them:

                 prey = circle      prey = corners
  pred = circle   in-distribution    OOD prey
  pred = corners  OOD prey           in-distribution

Each prey is evaluated in its NATIVE env (the layout it was trained for), so
its behaviour is sensible. The predator never observes resources, so within a
column the env and prey are held fixed and only the predator policy changes —
a clean isolation of predator brittleness. If the in-distribution predator
(diagonal) catches the prey more often than the OOD predator (off-diagonal),
the co-trained predator is brittle to an opponent strategy it never saw.

Metric: captures / episode = predator return / 10 (predators get +10 per
collision, shared across the team).

Usage:
    python src/brittleness_crossplay.py --algorithm mappo --seeds 3 --eps 200
    python src/brittleness_crossplay.py --algorithm iql   --seeds 3 --eps 200
"""
import argparse
import os
import sys

import jax
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_trajectory_dataset_resources as G
from generate_trajectory_dataset_resources import (
    split_teams, _make_env, _params_path,
    MLPQNetwork, MLPQOppNetwork, ActorLogits, HIDDEN, NUM_STEPS,
)
from jaxmarl.wrappers.baselines import (
    CTRolloutManager, MPELogWrapper, load_params,
)

LOGDIR = "logs/MPE_simple_tag_v3"
PLOTDIR = "plots"
PLACEMENTS = ["circle", "corners"]


def _build_nets(algorithm, teams, action_dim):
    if algorithm == "iql":
        return {t: MLPQNetwork(action_dim=action_dim, hidden_dim=HIDDEN) for t in teams}
    if algorithm == "oa_iql":
        return {
            "pred": MLPQOppNetwork(action_dim=action_dim, hidden_dim=HIDDEN,
                                   opp_n_agents=len(teams["prey"])),
            "prey": MLPQOppNetwork(action_dim=action_dim, hidden_dim=HIDDEN,
                                   opp_n_agents=len(teams["pred"])),
        }
    return {t: ActorLogits(action_dim=action_dim, hidden_dim=HIDDEN) for t in teams}


def crossplay_cell(algorithm, pred_placement, prey_placement, eval_placement,
                   seed_idx, num_eps, rng, ckpt_suffix=""):
    """Run num_eps greedy episodes with pred and prey loaded independently.

    Returns (captures_per_ep, pred_return, prey_return) averaged over episodes.
    """
    base_env = _make_env(eval_placement)
    teams = split_teams(base_env)
    env = MPELogWrapper(base_env)
    wrapped = CTRolloutManager(env, batch_size=num_eps)
    pred_names = list(teams["pred"])
    prey_name = teams["prey"][0]
    action_dim = wrapped.max_action_space
    max_obs = max(base_env.observation_space(a).shape[0] for a in base_env.agents)

    nets = _build_nets(algorithm, teams, action_dim)
    params = {
        "pred": load_params(_params_path(algorithm, pred_placement + ckpt_suffix, "pred", seed_idx)),
        "prey": load_params(_params_path(algorithm, prey_placement + ckpt_suffix, "prey", seed_idx)),
    }

    rng, k = jax.random.split(rng)
    obs, env_state = wrapped.batch_reset(k)
    pred_rew_tot = np.zeros(num_eps)
    prey_rew_tot = np.zeros(num_eps)

    for _ in range(NUM_STEPS):
        rng, ks = jax.random.split(rng)
        valid = wrapped.get_valid_actions(env_state.env_state)
        acts = {}
        for t in teams:
            for a in teams[t]:
                if algorithm == "mappo":
                    logits = nets[t].apply(params[t], obs[a][:, :max_obs])
                    acts[a] = logits.argmax(-1).astype("int32")
                elif algorithm == "oa_iql":
                    q, _ = nets[t].apply(params[t], obs[a])
                    q = q - (1 - valid[a]) * 1e10
                    acts[a] = q.argmax(-1).astype("int32")
                else:
                    q = nets[t].apply(params[t], obs[a])
                    q = q - (1 - valid[a]) * 1e10
                    acts[a] = q.argmax(-1).astype("int32")
        obs, env_state, rewards, dones, _ = wrapped.batch_step(ks, env_state, acts)
        pred_rew_tot += np.asarray(rewards[pred_names[0]])   # shared across pred team
        prey_rew_tot += np.asarray(rewards[prey_name])

    # captures/ep: predators get +10 per collision (shared). Episodes auto-reset
    # every 25 steps, so this is captures summed over the ~2 episodes in NUM_STEPS;
    # report per-25-step-episode by dividing by (NUM_STEPS/25).
    n_eps_in_traj = NUM_STEPS / 25.0
    captures = pred_rew_tot / 10.0 / n_eps_in_traj
    return float(captures.mean()), float(captures.std()), \
        float(pred_rew_tot.mean()), float(prey_rew_tot.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", default="mappo", choices=["iql", "oa_iql", "mappo"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--eps", type=int, default=200)
    p.add_argument("--ckpt_suffix", default="")
    p.add_argument("--pred_max_speed", type=float, default=None)
    p.add_argument("--pred_accel", type=float, default=None)
    p.add_argument("--collect_reward", type=float, default=5.0)
    p.add_argument("--tag", default="")
    args = p.parse_args()

    G.PRED_MAX_SPEED = args.pred_max_speed
    G.PRED_ACCEL = args.pred_accel
    G.COLLECT_REWARD = args.collect_reward
    tag = args.tag or args.algorithm

    os.makedirs(PLOTDIR, exist_ok=True)
    cap = np.zeros((2, 2, args.seeds))      # [pred, prey, seed] captures/ep
    for i, pp in enumerate(PLACEMENTS):           # predator origin
        for j, yp in enumerate(PLACEMENTS):       # prey origin
            for s in range(args.seeds):
                rng = jax.random.PRNGKey(1000 + s)
                # prey evaluated in its native env
                c, _, predr, preyr = crossplay_cell(
                    args.algorithm, pp, yp, yp, s, args.eps, rng, args.ckpt_suffix)
                cap[i, j, s] = c
            kind = "in-dist " if i == j else "OOD prey"
            print(f"  pred={pp:7s} vs prey={yp:7s} [{kind}]  "
                  f"captures/ep = {cap[i,j].mean():.2f} ± {cap[i,j].std():.2f}", flush=True)

    mean = cap.mean(-1)
    std = cap.std(-1)

    # In-distribution (diagonal) vs OOD (off-diagonal), column-wise control:
    # within each prey column the env+prey are fixed; only the predator differs.
    indist = np.array([mean[0, 0], mean[1, 1]])
    ood = np.array([mean[1, 0], mean[0, 1]])   # circle-prey vs corners-pred; corners-prey vs circle-pred
    print(f"\nIn-distribution predator captures/ep: {indist.mean():.2f}")
    print(f"OOD predator captures/ep:             {ood.mean():.2f}")
    drop = 100 * (indist.mean() - ood.mean()) / indist.mean()
    print(f"Relative drop when predator faces OOD prey: {drop:.0f}%")

    np.savez(os.path.join(LOGDIR, f"brittleness_{tag}.npz"),
             cap_mean=mean, cap_std=std, placements=np.array(PLACEMENTS, dtype="<U8"),
             indist=indist, ood=ood, drop=drop)

    # ---- figure ---- #
    fig, (axm, axb) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1.05, 1]})
    im = axm.imshow(mean, cmap="viridis", vmin=0)
    axm.set_xticks([0, 1]); axm.set_xticklabels([f"{p}-prey" for p in PLACEMENTS])
    axm.set_yticks([0, 1]); axm.set_yticklabels([f"{p}-pred" for p in PLACEMENTS])
    axm.set_xlabel("prey policy (native env)")
    axm.set_ylabel("predator policy")
    axm.set_title(f"{args.algorithm.upper()} captures / episode\n"
                  "diagonal = in-distribution, off-diagonal = OOD prey")
    for i in range(2):
        for j in range(2):
            edge = "in-dist" if i == j else "OOD"
            axm.text(j, i, f"{mean[i,j]:.2f}\n±{std[i,j]:.2f}\n({edge})",
                     ha="center", va="center",
                     color="white" if mean[i, j] < mean.max() * 0.6 else "black",
                     fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=axm, fraction=0.046, pad=0.04)

    axb.bar(["in-distribution\npredator", "OOD\npredator"],
            [indist.mean(), ood.mean()],
            yerr=[indist.std(), ood.std()], capsize=6,
            color=["#1f7a4e", "#b44233"])
    axb.set_ylabel("captures / episode")
    axb.set_title(f"Predator brittleness to OOD prey\n"
                  f"{drop:.0f}% fewer captures vs an unseen prey strategy")
    for i, v in enumerate([indist.mean(), ood.mean()]):
        axb.text(i, v + 0.03, f"{v:.2f}", ha="center", fontweight="bold")
    axb.grid(alpha=0.3, axis="y")

    fig.suptitle("Deliverable 3 — co-trained predators are brittle to "
                 "out-of-distribution opponents", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(PLOTDIR, f"brittleness_{tag}.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
