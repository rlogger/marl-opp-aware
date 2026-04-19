"""Exp 2 behavior mining: show that CW/CCW shaping produces *stereotyped*
prey motion that explains why shaping actually REDUCES prey return.

For each static variant {baseline, cw, ccw} and each seed {0,1,2}:
- Roll out NUM_EPS episodes of NUM_STEPS steps in the static env used for training
- Record the prey's position every step

Then compute:
1. 2D prey-position density (heatmap): obstacles at (+/-0.5, +/-0.5) visible as
   dwell zones for baseline vs orbital bands for CW/CCW.
2. Signed angular displacement distribution (per step, relative to nearest
   obstacle). Baseline: symmetric around 0. CW: negative-skewed. CCW:
   positive-skewed.
3. Mean captures/ep (sanity check that shaping *hurt* the prey).

Uses RNN net because static variants were trained with iql_teams.py (HS=64).
"""
import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iql_teams import RNNQNetwork, ScannedRNN, split_teams
from simple_tag_static import SimpleTagStaticMPE

from jaxmarl.wrappers.baselines import load_params, CTRolloutManager, MPELogWrapper


VARIANTS = {
    "baseline": dict(shape_coef=0.0, shape_dir="ccw"),
    "cw":        dict(shape_coef=0.1, shape_dir="cw"),
    "ccw":       dict(shape_coef=0.1, shape_dir="ccw"),
}
OBSTACLES = [[0.5, 0.5], [-0.5, -0.5]]
SEEDS = [0, 1, 2]
NUM_EPS = 64
NUM_STEPS = 32
HIDDEN = 64

LOGDIR = "logs/MPE_simple_tag_v3"
OUTDIR = "plots"


def load_variant_params(variant_key, seed_idx):
    alg_name = {"baseline": "iql_teams_static_baseline",
                "cw": "iql_teams_static_cw",
                "ccw": "iql_teams_static_ccw"}[variant_key]
    pred = load_params(f"{LOGDIR}/{alg_name}_MPE_simple_tag_v3_pred_seed0_vmap{seed_idx}.safetensors")
    prey = load_params(f"{LOGDIR}/{alg_name}_MPE_simple_tag_v3_prey_seed0_vmap{seed_idx}.safetensors")
    return pred, prey


def rollout_prey_trajectories(variant_key, seed_idx, num_eps=NUM_EPS, num_steps=NUM_STEPS, rng_seed=0):
    """Return (prey_positions, pred_returns_per_ep).
    prey_positions: (num_eps, num_steps+1, 2) ndarray of prey (x,y).
    """
    cfg = VARIANTS[variant_key]
    base_env = SimpleTagStaticMPE(obstacle_positions=OBSTACLES,
                                  shape_coef=cfg["shape_coef"],
                                  shape_direction=cfg["shape_dir"])
    teams = split_teams(base_env)
    env = MPELogWrapper(base_env)
    wrapped = CTRolloutManager(env, batch_size=num_eps)
    prey_name = teams["prey"][0]
    prey_idx = base_env.agents.index(prey_name)

    nets = {t: RNNQNetwork(action_dim=wrapped.max_action_space, hidden_dim=HIDDEN) for t in teams}
    pred_params, prey_params = load_variant_params(variant_key, seed_idx)
    params = {"pred": pred_params, "prey": prey_params}

    rng = jax.random.PRNGKey(rng_seed + seed_idx * 101)
    rng, key_reset = jax.random.split(rng)
    obs, env_state = wrapped.batch_reset(key_reset)

    # Per-team RNN hidden state: shape (n_agents, batch, hidden)
    hs = {t: ScannedRNN.initialize_carry(HIDDEN, len(teams[t]), num_eps) for t in teams}
    dones = {a: jnp.zeros(num_eps, dtype=bool) for a in env.agents + ["__all__"]}

    prey_positions = [np.asarray(env_state.env_state.p_pos[:, prey_idx, :])]
    pred_returns = np.zeros(num_eps, dtype=np.float32)

    for _ in range(num_steps):
        rng, key_step = jax.random.split(rng)
        valid = wrapped.get_valid_actions(env_state.env_state)
        actions = {}
        for t in teams:
            ags = teams[t]
            _obs = jnp.stack([obs[a] for a in ags], axis=0)[:, None]  # (n_ag, 1, batch, obs_dim)?
            # actually needs (n_ag, time=1, batch, obs) — check RNNQNetwork signature
            _dn = jnp.stack([dones[a] for a in ags], axis=0)[:, None]  # (n_ag, 1, batch)
            new_hs, q = jax.vmap(nets[t].apply, in_axes=(None, 0, 0, 0))(
                params[t], hs[t], _obs, _dn
            )
            q = q.squeeze(axis=1)  # (n_ag, batch, action_dim)
            va = jnp.stack([valid[a] for a in ags], axis=0)  # (n_ag, batch, a_dim)
            q = q - (1 - va) * 1e10
            a_team = jnp.argmax(q, axis=-1)  # (n_ag, batch)
            hs[t] = new_hs
            for i, name in enumerate(ags):
                actions[name] = a_team[i]
        obs, env_state, rewards, new_dones, infos = wrapped.batch_step(key_step, env_state, actions)
        dones = new_dones
        prey_positions.append(np.asarray(env_state.env_state.p_pos[:, prey_idx, :]))
        # Accumulate per-env pred return (sum over pred agents)
        for a in teams["pred"]:
            pred_returns += np.asarray(rewards[a])

    arr = np.stack(prey_positions, axis=1)  # (num_eps, num_steps+1, 2)
    return arr, pred_returns


def compute_angular_deltas(prey_pos, obstacles=np.array(OBSTACLES)):
    """Signed angular change per step relative to *nearest* obstacle.
    prey_pos: (num_eps, T+1, 2).  Returns: (num_eps * T,) flat array.
    Positive = CCW, negative = CW.
    """
    prev = prey_pos[:, :-1]   # (E, T, 2)
    nxt = prey_pos[:, 1:]     # (E, T, 2)
    # For each step pick the obstacle nearest to prev
    d_prev = np.linalg.norm(prev[:, :, None, :] - obstacles[None, None, :, :], axis=-1)  # (E,T,L)
    near = d_prev.argmin(axis=-1)  # (E,T)
    obs_chosen = obstacles[near]   # (E,T,2)
    vp = prev - obs_chosen
    vn = nxt - obs_chosen
    # Signed angle from vp to vn
    cross = vp[..., 0] * vn[..., 1] - vp[..., 1] * vn[..., 0]
    dot = vp[..., 0] * vn[..., 0] + vp[..., 1] * vn[..., 1]
    ang = np.arctan2(cross, dot)  # (E, T) — signed radians
    return ang.reshape(-1)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    all_positions = {}
    all_angles = {}
    all_pred_ret = {}
    for v in VARIANTS:
        eps_positions = []
        eps_pred_ret = []
        for s in SEEDS:
            print(f"rolling out {v} seed={s} ...", flush=True)
            pos, pr = rollout_prey_trajectories(v, s)
            eps_positions.append(pos)
            eps_pred_ret.append(pr)
        pos_all = np.concatenate(eps_positions, axis=0)  # (3*NUM_EPS, T+1, 2)
        pr_all = np.concatenate(eps_pred_ret, axis=0)
        all_positions[v] = pos_all
        all_angles[v] = compute_angular_deltas(pos_all)
        all_pred_ret[v] = pr_all
        print(f"  => {pos_all.shape[0]} episodes, mean pred_return={pr_all.mean():+.2f}")

    # ==== Figure: 2×3 panel ====
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for j, v in enumerate(["baseline", "cw", "ccw"]):
        pos = all_positions[v]
        ax = axes[0, j]
        ax.hist2d(pos[:, :, 0].ravel(), pos[:, :, 1].ravel(), bins=40,
                  range=[[-1.4, 1.4], [-1.4, 1.4]], cmap="magma")
        # overlay obstacles
        for ox, oy in OBSTACLES:
            circ = plt.Circle((ox, oy), 0.15, color="cyan", fill=False, lw=1.5)
            ax.add_patch(circ)
        ax.set_aspect("equal")
        ax.set_title(f"prey position density  |  {v}")
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])

        ax = axes[1, j]
        ang = all_angles[v]
        ax.hist(ang, bins=60, range=(-0.8, 0.8), color="tab:green", alpha=0.8)
        ax.axvline(0, color="k", lw=0.8, alpha=0.6)
        ax.axvline(ang.mean(), color="red", lw=1.3, ls="--", label=f"mean={ang.mean():+.3f} rad/step")
        ax.set_title(f"signed angular step  |  {v}")
        ax.set_xlabel("Δangle (rad, >0 = CCW)"); ax.set_ylabel("count")
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Exp 2: shaping induces stereotyped orbital motion in prey  |  3 seeds × 64 eps")
    fig.tight_layout()
    out = f"{OUTDIR}/exp2_prey_behavior.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")

    # Print summary numbers
    print("\n=== Mean signed angular step (radians) ===")
    for v in VARIANTS:
        ang = all_angles[v]
        pr = all_pred_ret[v]
        print(f"{v:10s}  mean(delta_angle)={ang.mean():+.4f}  std={ang.std():.4f}  "
              f"|  pred_return={pr.mean():+.2f}±{pr.std():.2f}")

    np.savez_compressed(
        f"{LOGDIR}/exp2_behavior.npz",
        **{f"{v}_positions": all_positions[v] for v in VARIANTS},
        **{f"{v}_angles": all_angles[v] for v in VARIANTS},
        **{f"{v}_pred_return": all_pred_ret[v] for v in VARIANTS},
    )


if __name__ == "__main__":
    main()
