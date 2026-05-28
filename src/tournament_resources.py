"""
Cross-play tournament for the resource environment.

4x4 matrix comparing IQL, OA-IQL (greedy and planning), and MAPPO:

              prey=IQL | prey=OA | prey=OA-plan | prey=MAPPO
  pred=IQL   |        |        |             |
  pred=OA    |        |        |             |
  pred=OA-plan|       |        |             |
  pred=MAPPO |        |        |             |

Each cell = mean +/- std over SEEDS, where each seed runs NUM_EPS eval
episodes (32-step horizon).  Metrics: predator return, prey return,
captures/episode.

The environment placement mode is configurable (--placement); default is
"random" (50/50 circle/corners).  All four policy variants are loaded from
checkpoints trained on that same placement.

Usage:
    python src/tournament_resources.py --placement random --seeds 3 --eps 100 --K 5 --H 3
"""
import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from jaxmarl.wrappers.baselines import MPELogWrapper, CTRolloutManager, load_params

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simple_tag_resources import SimpleTagResourcesMPE
from planning_eval import (
    make_plan_team_actions,
    make_greedy_team_actions,
    _split_teams,
)


LOGDIR    = "logs/MPE_simple_tag_v3"
OBSTACLES = [[0.5, 0.5], [-0.5, -0.5]]

POLICY_NAMES = ["IQL-greedy", "OA-greedy", "OA-plan", "MAPPO"]


# ------------------------------------------------------------------ #
# MAPPO actor (logits only, no distrax)
# ------------------------------------------------------------------ #

class ActorLogits(nn.Module):
    """MAPPO Actor returning raw logits.
    Param-compatible with mappo_teams_mlp.Actor -- identical Dense structure."""
    action_dim: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                        bias_init=constant(0.0))(x)


# ------------------------------------------------------------------ #
# Checkpoint loading
# ------------------------------------------------------------------ #

def _load_params_for(team, variant, placement, seed_idx):
    """Load checkpoint params for a given policy variant + placement."""
    if variant == "IQL-greedy":
        alg = f"iql_teams_resources_{placement}"
        path = f"{LOGDIR}/{alg}_MPE_simple_tag_v3_{team}_seed0_vmap{seed_idx}.safetensors"
    elif variant in ("OA-greedy", "OA-plan"):
        alg = f"iql_teams_oa_resources_{placement}"
        path = f"{LOGDIR}/{alg}_MPE_simple_tag_v3_{team}_seed0_vmap{seed_idx}.safetensors"
    elif variant == "MAPPO":
        alg = f"mappo_teams_resources_{placement}"
        path = (f"{LOGDIR}/{alg}_MPE_simple_tag_v3_{team}"
                f"_actor_seed0_vmap{seed_idx}.safetensors")
    else:
        raise ValueError(variant)
    return load_params(path)


# ------------------------------------------------------------------ #
# MAPPO greedy selector
# ------------------------------------------------------------------ #

def make_mappo_greedy(wrapped_env, team_name, action_dim=5, hidden_dim=128):
    """Greedy MAPPO action selector: argmax over actor logits.

    MAPPO was trained on pad_obs (no CTRolloutManager one-hot), so we strip
    the trailing agent-id dims from the CTRolloutManager observations before
    feeding the actor.
    """
    teams = _split_teams(wrapped_env)
    team_agents = teams[team_name]

    # Compute max raw obs dim (without one-hot)
    inner_env = wrapped_env._env  # MPELogWrapper
    base_env  = inner_env._env    # SimpleTagResourcesMPE
    obs_sizes = {a: base_env.observation_space(a).shape[0]
                 for a in base_env.agents}
    max_obs = max(obs_sizes.values())

    actor = ActorLogits(action_dim=action_dim, hidden_dim=hidden_dim)

    def greedy_team(params, obs_dict, env_state, rng):
        out = {}
        for ag in team_agents:
            logits = actor.apply(params, obs_dict[ag][:max_obs])
            out[ag] = jnp.argmax(logits).astype(jnp.int32)
        return out

    return greedy_team


# ------------------------------------------------------------------ #
# Selector factory
# ------------------------------------------------------------------ #

def make_selector(test_env, team, variant, K, H, gamma):
    """Return a `select(params, obs_dict, env_state, rng) -> action_dict`."""
    if variant == "OA-plan":
        return make_plan_team_actions(test_env, team, K=K, H=H, gamma=gamma)
    elif variant == "MAPPO":
        return make_mappo_greedy(test_env, team)
    else:
        has_opp_head = (variant == "OA-greedy")
        return make_greedy_team_actions(test_env, team, has_opp_head=has_opp_head)


# ------------------------------------------------------------------ #
# Evaluation
# ------------------------------------------------------------------ #

def run_cell(env_base, test_env, pred_sel, prey_sel,
             pred_params, prey_params, rng, num_steps):
    """Run one eval rollout of length `num_steps` in `test_env`.
    Returns per-episode metrics via MPELogWrapper infos."""
    teams = _split_teams(env_base)

    vmap_pred = jax.vmap(pred_sel, in_axes=(None, 0, 0, 0))
    vmap_prey = jax.vmap(prey_sel, in_axes=(None, 0, 0, 0))

    rng, rng_reset = jax.random.split(rng)
    init_obs, env_state = test_env.batch_reset(rng_reset)

    def _step(carry, _):
        obs_d, env_state, rng = carry
        rng, rng_pred, rng_prey, rng_step = jax.random.split(rng, 4)
        rngs_pred = jax.random.split(rng_pred, test_env.batch_size)
        rngs_prey = jax.random.split(rng_prey, test_env.batch_size)

        pred_actions = vmap_pred(pred_params, obs_d, env_state, rngs_pred)
        prey_actions = vmap_prey(prey_params, obs_d, env_state, rngs_prey)
        actions = {**pred_actions, **prey_actions}

        new_obs, new_env_state, rewards, new_dones, infos = test_env.batch_step(
            rng_step, env_state, actions)
        return (new_obs, new_env_state, rng), (rewards, new_dones, infos)

    init_carry = (init_obs, env_state, rng)
    _, (rewards, dones, infos) = jax.lax.scan(
        _step, init_carry, None, num_steps)
    return rewards, dones, infos


def summarize(infos, env, teams):
    """Per-team mean return over completed episodes.

    captures_per_ep = pred_return / 10 (each capture gives predator +10).
    In the resource env, prey_return reflects both capture penalties (-10)
    and resource bonuses (+5), so it is not simply -pred_return.
    """
    out = {}
    for t in ["pred", "prey"]:
        idxs = jnp.array([env.agents.index(a) for a in teams[t]])
        mask = infos["returned_episode"][..., idxs]
        rets = jnp.where(mask,
                         infos["returned_episode_returns"][..., idxs],
                         jnp.nan)
        out[f"{t}_return"] = float(jnp.nanmean(rets))
    out["captures_per_ep"] = out["pred_return"] / 10.0
    return out


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser(
        description="Cross-play tournament on the resource environment.")
    ap.add_argument("--placement", default="random",
                    choices=["circle", "corners", "random"],
                    help="Resource placement mode for the tournament env "
                         "and checkpoint selection (default: random).")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eps", type=int, default=100,
                    help="Parallel eval episodes per seed per cell.")
    ap.add_argument("--num_steps", type=int, default=32)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--master_seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None,
                    help="Output npz path (default: auto from placement).")
    args = ap.parse_args()

    if args.out is None:
        args.out = f"{LOGDIR}/tournament_resources_{args.placement}.npz"

    # Build resource env
    base_env = SimpleTagResourcesMPE(
        num_resources=4, placement=args.placement,
        collect_radius=0.15, collect_reward=5.0,
        circle_radius=0.6, corner_offset=0.8,
        obstacle_positions=OBSTACLES,
    )
    env      = MPELogWrapper(base_env)
    test_env = CTRolloutManager(env, batch_size=args.eps)
    teams    = _split_teams(env)
    master_rng = jax.random.PRNGKey(args.master_seed)

    print(f"Tournament: placement={args.placement}, seeds={args.seeds}, "
          f"eps={args.eps}, steps={args.num_steps}")
    print(f"Policies: {POLICY_NAMES}")

    # Precompute selectors
    pred_sels = {v: make_selector(test_env, "pred", v,
                                  K=args.K, H=args.H, gamma=args.gamma)
                 for v in POLICY_NAMES}
    prey_sels = {v: make_selector(test_env, "prey", v,
                                  K=args.K, H=args.H, gamma=args.gamma)
                 for v in POLICY_NAMES}

    # Storage: results[pred_variant][prey_variant] = list of dicts
    results = {p: {q: [] for q in POLICY_NAMES} for p in POLICY_NAMES}

    t0 = time.time()
    for pred_var in POLICY_NAMES:
        for prey_var in POLICY_NAMES:
            cell_t0 = time.time()
            for seed_idx in range(args.seeds):
                pred_params = _load_params_for("pred", pred_var,
                                               args.placement, seed_idx)
                prey_params = _load_params_for("prey", prey_var,
                                               args.placement, seed_idx)
                master_rng, rng = jax.random.split(master_rng)
                rewards, dones, infos = run_cell(
                    env, test_env,
                    pred_sels[pred_var], prey_sels[prey_var],
                    pred_params, prey_params, rng, args.num_steps)
                results[pred_var][prey_var].append(
                    summarize(infos, env, teams))

            dt = time.time() - cell_t0
            cell_vals = results[pred_var][prey_var]
            pred_ret = np.array([c["pred_return"] for c in cell_vals])
            print(f"  pred={pred_var:<11} prey={prey_var:<11} "
                  f"pred={pred_ret.mean():+7.2f}+/-{pred_ret.std():.2f}  "
                  f"({dt:.1f}s)")

    total_t = time.time() - t0
    print(f"\nTotal wall-clock: {total_t:.1f}s")

    # Dump 4x4 matrices to NPZ
    n = len(POLICY_NAMES)
    metrics = ["pred_return", "prey_return", "captures_per_ep"]
    out = {}
    for m in metrics:
        mat_mean = np.zeros((n, n))
        mat_std  = np.zeros((n, n))
        for i, p in enumerate(POLICY_NAMES):
            for j, q in enumerate(POLICY_NAMES):
                vals = np.array([c[m] for c in results[p][q]])
                mat_mean[i, j] = vals.mean()
                mat_std[i, j]  = vals.std()
        out[f"{m}_mean"] = mat_mean
        out[f"{m}_std"]  = mat_std
    out["policy_names"] = np.array(POLICY_NAMES)
    out["num_seeds"]    = np.array([args.seeds])
    out["num_eps"]      = np.array([args.eps])
    out["placement"]    = np.array([args.placement])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
