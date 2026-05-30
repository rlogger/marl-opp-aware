"""Part 1 of the proposal: a latent-conditioned opponent (prey) policy via BC.

The status slide's next step: "once we have a VAE we're happy with, train BC
conditioned on the VAE latent variable." Exp 5 (P2) validated that an
unsupervised occupancy-VAE recovers the prey's strategy into a latent z. Here we

  1. roll out the circle- and corners-specialist prey, logging (obs, action) and
     the full trajectory;
  2. encode each trajectory with the occupancy-VAE -> per-trajectory z, and the
     class-mean codes z_circle, z_corners;
  3. train a behaviour-cloning policy  pi_psi(a | obs, z)  on (obs, z, action);
  4. test latent control: roll pi_psi out with z held at z_circle vs z_corners
     and check it reproduces the corresponding route.

A successful Part 1 is a single policy that imitates either specialist purely by
swapping its latent input -- the opponent model the planner (Part 2) will sample.

Usage:
    python src/part1_latent_bc.py --algorithm iql
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
from flax.training.train_state import TrainState
import optax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_trajectory_dataset_resources as G
from generate_trajectory_dataset_resources import (
    rollout_one_checkpoint, _make_env, split_teams, _params_path,
    MLPQNetwork, MLPQOppNetwork, ActorLogits, HIDDEN,
)
from exp5b_encoder_length import occupancy, train_vae_get_z
from exp4_vae_separation import probe_acc
from jaxmarl.wrappers.baselines import CTRolloutManager, MPELogWrapper, load_params

LOGDIR = "logs/MPE_simple_tag_v3"
PLOTDIR = "plots"
PLACEMENTS = ["circle", "corners"]
SEEDS = [0, 1, 2]
WP = dict(pred_max_speed=0.6, pred_accel=1.5, collect_reward=10.0)
NUM_EPS = 80
STEPS = 100
Z_DIM = 3
COL = {"circle": "#0d6e7a", "corners": "#b85c10"}


def set_wp():
    G.PRED_MAX_SPEED = WP["pred_max_speed"]
    G.PRED_ACCEL = WP["pred_accel"]
    G.COLLECT_REWARD = WP["collect_reward"]
    G.NUM_STEPS = STEPS


# --------------------------------------------------------------------------- #
# BC policy
# --------------------------------------------------------------------------- #

class BCPolicy(nn.Module):
    action_dim: int = 5
    hidden: int = 128

    @nn.compact
    def __call__(self, obs, z):
        x = jnp.concatenate([obs, z], axis=-1)
        for _ in range(2):
            x = nn.relu(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)),
                                 bias_init=constant(0.0))(x))
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                        bias_init=constant(0.0))(x)


def collect(algorithm):
    """Roll specialists logging (obs, action) and trajectories. Returns pooled
    obs (M,O), actions (M,), per-sample z-trajectory index, and per-traj arrays."""
    set_wp()
    obs_all, act_all, traj_idx, traj_pos, traj_lab = [], [], [], [], []
    tcount = 0
    for k, placement in enumerate(PLACEMENTS):
        for seed in SEEDS:
            rng = jax.random.PRNGKey(4242 + seed)
            d = rollout_one_checkpoint(algorithm, placement + "_wp", seed,
                                       NUM_EPS, placement, rng, log_obs=True)
            obs = d["prey_obs"]            # (N, T, O)
            act = d["actions"]             # (N, T)
            pos = d["positions"]           # (N, T+1, 2)
            N, T, O = obs.shape
            for e in range(N):
                obs_all.append(obs[e]); act_all.append(act[e])
                traj_idx.append(np.full(T, tcount, np.int32))
                traj_pos.append(pos[e]); traj_lab.append(k)
                tcount += 1
            print(f"  [{algorithm}] {placement} seed{seed}: {N} eps, obs {O}-d",
                  flush=True)
    obs_all = np.concatenate(obs_all).astype(np.float32)
    act_all = np.concatenate(act_all).astype(np.int32)
    traj_idx = np.concatenate(traj_idx)
    traj_pos = np.stack(traj_pos)          # (n_traj, T+1, 2)
    traj_lab = np.array(traj_lab, np.int32)
    return obs_all, act_all, traj_idx, traj_pos, traj_lab


def train_bc(obs, z_per_sample, act, steps=8000, seed=0):
    rng = jax.random.PRNGKey(seed)
    O = obs.shape[1]
    net = BCPolicy()
    p = net.init(rng, jnp.zeros((1, O)), jnp.zeros((1, Z_DIM)))
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))
    st = TrainState.create(apply_fn=net.apply, params=p, tx=tx)

    X = jnp.asarray(obs); Z = jnp.asarray(z_per_sample); Y = jnp.asarray(act)
    n = X.shape[0]
    nval = n // 10
    perm = np.array(jax.random.permutation(jax.random.PRNGKey(1), n))
    vi, ti = perm[:nval], perm[nval:]

    @jax.jit
    def step(st, idx):
        def loss_fn(pp):
            logits = net.apply(pp, X[idx], Z[idx])
            return optax.softmax_cross_entropy_with_integer_labels(logits, Y[idx]).mean()
        l, g = jax.value_and_grad(loss_fn)(st.params)
        return st.apply_gradients(grads=g), l

    @jax.jit
    def acc(pp, idx):
        return (net.apply(pp, X[idx], Z[idx]).argmax(-1) == Y[idx]).mean()

    ti_j = jnp.asarray(ti); vi_j = jnp.asarray(vi)
    for s in range(steps):
        rng, bk = jax.random.split(rng)
        idx = ti_j[jax.random.choice(bk, ti.shape[0], (256,), replace=False)]
        st, l = step(st, idx)
        if s % 2000 == 0 or s == steps - 1:
            print(f"    bc step {s:5d}  loss {float(l):.3f}  "
                  f"val-acc {float(acc(st.params, vi_j)):.3f}", flush=True)
    return net, st.params, float(acc(st.params, vi_j)), float(acc(st.params, ti_j))


# --------------------------------------------------------------------------- #
# Rollout the BC prey against specialist predators, latent z held fixed
# --------------------------------------------------------------------------- #

def bc_rollout(bc_net, bc_params, z_vec, algorithm, pred_placement,
               env_placement, seed_idx, n_eps=120, steps=100, eval_seed=7):
    set_wp()
    base = _make_env(env_placement)
    teams = split_teams(base)
    wrapped = CTRolloutManager(MPELogWrapper(base), batch_size=n_eps)
    prey = teams["prey"][0]; preds = list(teams["pred"])
    prey_i = base.agents.index(prey)
    action_dim = wrapped.max_action_space
    max_obs = max(base.observation_space(a).shape[0] for a in base.agents)

    if algorithm == "iql":
        pnet = MLPQNetwork(action_dim=action_dim, hidden_dim=HIDDEN)
    elif algorithm == "oa_iql":
        pnet = MLPQOppNetwork(action_dim=action_dim, hidden_dim=HIDDEN,
                              opp_n_agents=1)
    else:
        pnet = ActorLogits(action_dim=action_dim, hidden_dim=HIDDEN)
    pred_params = load_params(_params_path(algorithm, pred_placement + "_wp",
                                           "pred", seed_idx))
    z_b = jnp.broadcast_to(jnp.asarray(z_vec)[None], (n_eps, Z_DIM))

    rng = jax.random.PRNGKey(eval_seed)
    rng, kr = jax.random.split(rng)
    obs, est = wrapped.batch_reset(kr)
    pos = [np.asarray(est.env_state.p_pos[:, prey_i])]
    for _ in range(steps):
        rng, ks = jax.random.split(rng)
        valid = wrapped.get_valid_actions(est.env_state)
        acts = {}
        acts[prey] = bc_net.apply(bc_params, obs[prey], z_b).argmax(-1).astype(jnp.int32)
        for a in preds:
            if algorithm == "mappo":
                q = pnet.apply(pred_params, obs[a][:, :max_obs])
            elif algorithm == "oa_iql":
                q, _ = pnet.apply(pred_params, obs[a]); q = q - (1 - valid[a]) * 1e10
            else:
                q = pnet.apply(pred_params, obs[a]); q = q - (1 - valid[a]) * 1e10
            acts[a] = q.argmax(-1).astype(jnp.int32)
        obs, est, _, _, _ = wrapped.batch_step(ks, est, acts)
        pos.append(np.asarray(est.env_state.p_pos[:, prey_i]))
    return np.stack(pos, axis=1)              # (n_eps, steps+1, 2)


def route_probe(posA, posB):
    """Occupancy-probe accuracy separating two sets of rollouts (chance 0.5)."""
    occA = np.stack([occupancy(p[1:]) for p in posA])
    occB = np.stack([occupancy(p[1:]) for p in posB])
    X = np.concatenate([occA, occB]); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    y = np.r_[np.zeros(len(occA)), np.ones(len(occB))].astype(int)
    return probe_acc(X, y)


def density_panel(ax, pos, title, res_layout):
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    xy = pos[:, 1:].reshape(-1, 2)
    ax.hexbin(xy[:, 0], xy[:, 1], gridsize=30, cmap="magma", mincnt=1,
              extent=(-1.6, 1.6, -1.6, 1.6))
    for g in res_layout:
        ax.scatter(*g, s=40, facecolors="none", edgecolors="#39d0ff", linewidths=1.3)
    ax.set_title(title, fontsize=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", default="iql", choices=["iql", "oa_iql", "mappo"])
    args = ap.parse_args()
    os.makedirs(PLOTDIR, exist_ok=True)
    alg = args.algorithm

    # 1. data
    obs, act, traj_idx, traj_pos, traj_lab = collect(alg)

    # 2. encode each trajectory with the occupancy-VAE -> z; class means
    occ = np.stack([occupancy(p[1:]) for p in traj_pos])
    occ_std = (occ - occ.mean(0)) / (occ.std(0) + 1e-6)
    z_traj = train_vae_get_z(occ_std, latent=Z_DIM)          # (n_traj, Z_DIM)
    z_per_sample = z_traj[traj_idx]                           # (M, Z_DIM)
    z_circle = z_traj[traj_lab == 0].mean(0)
    z_corners = z_traj[traj_lab == 1].mean(0)
    sep = probe_acc((z_traj - z_traj.mean(0)) / (z_traj.std(0) + 1e-6), traj_lab)
    print(f"latent separates specialists at probe {sep:.2f}")

    # 3. BC
    net, params, val_acc, tr_acc = train_bc(obs, z_per_sample, act)
    maj = np.bincount(act).max() / len(act)
    print(f"BC action-match: val {val_acc:.2f}  train {tr_acc:.2f}  "
          f"(majority-action {maj:.2f})")

    # 4. latent control
    circ_layout = [(0.6 * np.cos(t), 0.6 * np.sin(t)) for t in
                   np.pi / 2 * np.arange(4)]
    corn_layout = [(sx * 0.8, sy * 0.8) for sx in (-1, 1) for sy in (-1, 1)]

    # matched: BC under z_circle in circle env, z_corners in corners env
    pos_cc = bc_rollout(net, params, z_circle, alg, "circle", "circle", 0)
    pos_kk = bc_rollout(net, params, z_corners, alg, "corners", "corners", 0)
    matched_sep = route_probe(pos_cc, pos_kk)

    # controlled: same env (circle) + circle predators, only z swapped
    pos_c_zc = bc_rollout(net, params, z_circle, alg, "circle", "circle", 0)
    pos_c_zk = bc_rollout(net, params, z_corners, alg, "circle", "circle", 0)
    control_sep = route_probe(pos_c_zc, pos_c_zk)
    print(f"matched-env route separation: {matched_sep:.2f}")
    print(f"latent-control (fixed circle env, swap z): {control_sep:.2f}")

    np.savez(os.path.join(LOGDIR, f"part1_latent_bc_{alg}.npz"),
             val_acc=val_acc, train_acc=tr_acc, majority=maj, latent_sep=sep,
             matched_sep=matched_sep, control_sep=control_sep,
             z_circle=z_circle, z_corners=z_corners)

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    density_panel(axes[0], pos_cc, "BC | z = z_circle  (circle env)", circ_layout)
    density_panel(axes[1], pos_kk, "BC | z = z_corners  (corners env)", corn_layout)
    ax = axes[2]
    bars = ax.bar(["BC action\nmatch", "majority\naction", "matched-env\nroute sep",
                   "latent control\n(fixed env)"],
                  [val_acc, maj, matched_sep, control_sep],
                  color=["#1f7a4e", "#999", "#0d6e7a", "#b85c10"])
    ax.axhline(0.5, ls="--", c="k", lw=1)
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy")
    ax.set_title("BC imitation + latent controllability")
    for b, v in zip(bars, [val_acc, maj, matched_sep, control_sep]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontweight="bold", fontsize=9)
    fig.suptitle(f"Part 1 — {alg.upper()} latent-conditioned BC opponent: one policy, "
                 f"two routes by swapping z", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(PLOTDIR, f"part1_latent_bc_{alg}.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
