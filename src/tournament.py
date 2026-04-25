"""
Cross-play tournament between IQL-MLP, OA-IQL-MLP (greedy), and OA-IQL-MLP (planning).

A 3x3 matrix, paired by seed:
                 prey=IQL-greedy | prey=OA-greedy | prey=OA-plan
  pred=IQL-greedy |   cell       |    cell        |   cell
  pred=OA-greedy  |   cell       |    cell        |   cell
  pred=OA-plan    |   cell       |    cell        |   cell

Each cell = mean +/- std over SEEDS, where each seed runs NUM_EPS eval episodes
(32-step horizon). Metrics: predator return, prey return, captures/episode.

Usage:
    python src/tournament.py --seeds 3 --eps 100 --K 5 --H 3
"""
import argparse
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl import make
from jaxmarl.wrappers.baselines import MPELogWrapper, CTRolloutManager, load_params

from planning_eval import (
    make_plan_team_actions,
    make_greedy_team_actions,
    _split_teams,
)


LOGDIR = "logs/MPE_simple_tag_v3"
ALG_IQL = "iql_teams_mlp"
ALG_OA = "iql_teams_oa_mlp"

POLICY_NAMES = ["IQL-greedy", "OA-greedy", "OA-plan"]


def _load_params_for(team: str, variant: str, seed_idx: int):
    """`variant` in {'IQL-greedy', 'OA-greedy', 'OA-plan'}."""
    alg = ALG_IQL if variant == "IQL-greedy" else ALG_OA
    path = f"{LOGDIR}/{alg}_MPE_simple_tag_v3_{team}_seed0_vmap{seed_idx}.safetensors"
    return load_params(path)


def make_selector(test_env, team: str, variant: str, K: int, H: int, gamma: float):
    """Return a `select(params, obs_dict, env_state, rng) -> action_dict`
    for the given team + variant. `test_env` is the CTRolloutManager."""
    if variant == "OA-plan":
        return make_plan_team_actions(test_env, team, K=K, H=H, gamma=gamma)
    has_opp_head = variant == "OA-greedy"
    return make_greedy_team_actions(test_env, team, has_opp_head=has_opp_head)


def run_cell(env_base, test_env, pred_sel, prey_sel,
             pred_params, prey_params, rng, num_steps: int):
    """Run one eval rollout of length `num_steps` in `test_env` (already has
    NUM_EPS parallel envs). Returns per-episode metrics."""
    teams = _split_teams(env_base)

    # vmap selectors across the batch of NUM_EPS parallel envs
    vmap_pred = jax.vmap(pred_sel, in_axes=(None, 0, 0, 0))
    vmap_prey = jax.vmap(prey_sel, in_axes=(None, 0, 0, 0))

    rng, rng_reset = jax.random.split(rng)
    init_obs, env_state = test_env.batch_reset(rng_reset)
    init_dones = {a: jnp.zeros((test_env.batch_size,), dtype=bool)
                  for a in env_base.agents + ["__all__"]}

    # Turn batched obs_dict and env_state.env_state into what per-episode selectors expect.
    # test_env.batch_reset returns env_state with .env_state being the per-env State (pytree with batch dim).

    def _step(carry, _):
        obs_d, env_state, dones, rng = carry
        rng, rng_pred, rng_prey, rng_step = jax.random.split(rng, 4)
        rngs_pred = jax.random.split(rng_pred, test_env.batch_size)
        rngs_prey = jax.random.split(rng_prey, test_env.batch_size)
        # wrapped_step expects the LogEnvState directly (not unwrapped).
        pred_actions = vmap_pred(pred_params, obs_d, env_state, rngs_pred)
        prey_actions = vmap_prey(prey_params, obs_d, env_state, rngs_prey)
        actions = {**pred_actions, **prey_actions}
        new_obs, new_env_state, rewards, new_dones, infos = test_env.batch_step(
            rng_step, env_state, actions
        )
        return (new_obs, new_env_state, new_dones, rng), (rewards, new_dones, infos)

    init_carry = (init_obs, env_state, init_dones, rng)
    _, (rewards, dones, infos) = jax.lax.scan(_step, init_carry, None, num_steps)
    return rewards, dones, infos


def summarize(infos, env, teams):
    """Collect per-team mean return over completed episodes."""
    out = {}
    for t in ["pred", "prey"]:
        idxs = jnp.array([env.agents.index(a) for a in teams[t]])
        mask = infos["returned_episode"][..., idxs]
        rets = jnp.where(mask, infos["returned_episode_returns"][..., idxs], jnp.nan)
        mean_ret = float(jnp.nanmean(rets))
        out[f"{t}_return"] = mean_ret
    # captures/ep = pred_return / 10
    out["captures_per_ep"] = out["pred_return"] / 10.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3,
                    help="Number of param seeds to average across per cell.")
    ap.add_argument("--eps", type=int, default=100,
                    help="Parallel eval episodes per seed per cell.")
    ap.add_argument("--num_steps", type=int, default=32)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--master_seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="logs/MPE_simple_tag_v3/tournament.npz")
    args = ap.parse_args()

    base_env = make("MPE_simple_tag_v3")
    env = MPELogWrapper(base_env)
    test_env = CTRolloutManager(env, batch_size=args.eps)
    teams = _split_teams(env)
    master_rng = jax.random.PRNGKey(args.master_seed)

    # Precompute selectors for every variant (selectors need the CTRolloutManager
    # for obs preprocessing / wrapped_step).
    pred_sels = {
        v: make_selector(test_env, "pred", v, K=args.K, H=args.H, gamma=args.gamma)
        for v in POLICY_NAMES
    }
    prey_sels = {
        v: make_selector(test_env, "prey", v, K=args.K, H=args.H, gamma=args.gamma)
        for v in POLICY_NAMES
    }

    # Storage: results[pred_variant][prey_variant] = list of dicts (one per seed)
    results = {p: {q: [] for q in POLICY_NAMES} for p in POLICY_NAMES}

    t0 = time.time()
    for pred_var in POLICY_NAMES:
        for prey_var in POLICY_NAMES:
            cell_t0 = time.time()
            for seed_idx in range(args.seeds):
                pred_params = _load_params_for("pred", pred_var, seed_idx)
                prey_params = _load_params_for("prey", prey_var, seed_idx)
                master_rng, rng = jax.random.split(master_rng)
                rewards, dones, infos = run_cell(
                    env, test_env,
                    pred_sels[pred_var], prey_sels[prey_var],
                    pred_params, prey_params, rng, args.num_steps
                )
                summary = summarize(infos, env, teams)
                results[pred_var][prey_var].append(summary)
            dt = time.time() - cell_t0
            # Mean across seeds for this cell
            cell_vals = results[pred_var][prey_var]
            pred_ret = np.array([c["pred_return"] for c in cell_vals])
            print(
                f"  cell pred={pred_var:<11} prey={prey_var:<11} "
                f"pred={pred_ret.mean():+6.2f}±{pred_ret.std():.2f} "
                f"({dt:.1f}s)"
            )
    total_t = time.time() - t0
    print(f"\nTotal tournament wall-clock: {total_t:.1f}s")

    # Dump to NPZ as stacked 3x3 arrays
    metrics = ["pred_return", "prey_return", "captures_per_ep"]
    out = {}
    for m in metrics:
        mat_mean = np.zeros((3, 3))
        mat_std = np.zeros((3, 3))
        for i, p in enumerate(POLICY_NAMES):
            for j, q in enumerate(POLICY_NAMES):
                vals = np.array([c[m] for c in results[p][q]])
                mat_mean[i, j] = vals.mean()
                mat_std[i, j] = vals.std()
        out[f"{m}_mean"] = mat_mean
        out[f"{m}_std"] = mat_std
    out["policy_names"] = np.array(POLICY_NAMES)
    out["num_seeds"] = np.array([args.seeds])
    out["num_eps"] = np.array([args.eps])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
