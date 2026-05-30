"""Generate prey trajectory datasets from trained MLP checkpoints on resource env.

Mirrors generate_trajectory_dataset.py but uses MLP checkpoints from Exp 4
(resource-augmented environment) instead of RNN checkpoints from Exp 2.

Two datasets per run:

  Reading A -- multi-checkpoint dataset.
    9 checkpoints: {circle, corners, random} placement x {seed 0,1,2}.
    NUM_EPS_A rollouts per checkpoint, T = NUM_STEPS steps each.
    Labelled by checkpoint id.

  Reading B -- single-checkpoint dataset.
    One checkpoint (default circle seed 0). NUM_EPS_B rollouts.
    Unlabelled. Use for single-policy multimodality analysis.

Select algorithm backbone via --algorithm: iql (default), oa_iql, or mappo.
All rollouts use the --eval_placement env (default: random) so that both
circle and corners layouts appear regardless of the training placement.

Outputs (.npz under logs/MPE_simple_tag_v3/):
  trajectory_dataset_resources_A.npz        (or _oa_iql_A / _mappo_A)
  trajectory_dataset_resources_B.npz

Schema extends the standard trajectory schema with resource data:
  positions       (N, T+1, 2)      float32   prey (x,y)
  pred_positions  (N, T+1, 3, 2)   float32   three predators (x,y)
  resource_pos    (N, T+1, 4, 2)   float32   resource positions
  collected       (N, T+1, 4)      bool      resource collected flags
  init_full       (N, n_ent, 2)    float32   reset positions all entities
  actions         (N, T)           int32     prey action
  rewards         (N, T)           float32   prey reward
  pred_actions    (N, T, 3)        int32     predator actions
  pred_rewards    (N, T, 3)        float32   predator rewards
  labels          (N,)             int32     checkpoint id (A) or -1 (B)
  checkpoints     (K,)             <U64      checkpoint name lookup

Usage:
    python src/generate_trajectory_dataset_resources.py --algorithm iql
    python src/generate_trajectory_dataset_resources.py --algorithm mappo --eval_placement circle
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simple_tag_resources import SimpleTagResourcesMPE
from jaxmarl.wrappers.baselines import CTRolloutManager, MPELogWrapper, load_params


LOGDIR     = "logs/MPE_simple_tag_v3"
NUM_STEPS  = 50
HIDDEN     = 128
OBSTACLES  = [[0.5, 0.5], [-0.5, -0.5]]

PLACEMENTS = ["circle", "corners", "random"]
SEEDS_A    = [0, 1, 2]
NUM_EPS_A  = 100   # per checkpoint -> 900 total
NUM_EPS_B  = 900


# ------------------------------------------------------------------ #
# Network architectures (for param loading -- no training logic)
# ------------------------------------------------------------------ #

class MLPQNetwork(nn.Module):
    """IQL MLP Q-network. Matches iql_teams_mlp.MLPQNetwork."""
    action_dim: int
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale),
                     bias_init=constant(0.0))(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale),
                     bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale),
                        bias_init=constant(0.0))(x)


class MLPQOppNetwork(nn.Module):
    """OA-IQL two-headed MLP. Matches iql_teams_oa_mlp.MLPQOppNetwork."""
    action_dim: int
    hidden_dim: int
    opp_n_agents: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale),
                     bias_init=constant(0.0))(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale),
                     bias_init=constant(0.0))(x)
        x = nn.relu(x)
        q = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale),
                     bias_init=constant(0.0))(x)
        opp = nn.Dense(self.opp_n_agents * self.action_dim,
                       kernel_init=orthogonal(self.init_scale),
                       bias_init=constant(0.0))(x)
        opp = opp.reshape(*opp.shape[:-1], self.opp_n_agents, self.action_dim)
        return q, opp


class ActorLogits(nn.Module):
    """MAPPO Actor returning raw logits (avoids distrax dependency).
    Param-compatible with mappo_teams_mlp.Actor -- same Dense structure."""
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
# Helpers
# ------------------------------------------------------------------ #

def split_teams(env):
    preds = [a for a in env.agents if a.startswith("adversary")]
    prey  = [a for a in env.agents if a.startswith("agent")]
    return {"pred": preds, "prey": prey}


def _params_path(algorithm, placement, team, seed_idx):
    """Checkpoint path matching the save convention in each trainer."""
    if algorithm == "iql":
        alg = f"iql_teams_resources_{placement}"
    elif algorithm == "oa_iql":
        alg = f"iql_teams_oa_resources_{placement}"
    elif algorithm == "mappo":
        alg = f"mappo_teams_resources_{placement}"
    else:
        raise ValueError(algorithm)

    base = f"{LOGDIR}/{alg}_MPE_simple_tag_v3_{team}"
    if algorithm == "mappo":
        # MAPPO saves actor and critic separately; we only need actor
        return f"{base}_actor_seed0_vmap{seed_idx}.safetensors"
    return f"{base}_seed0_vmap{seed_idx}.safetensors"


# Eval-env knobs. Set these (e.g. from an analysis script) so the rollout env
# matches how the checkpoints were trained — in particular weakened predators.
PRED_MAX_SPEED = None
PRED_ACCEL = None
COLLECT_REWARD = 5.0


def _make_env(placement):
    return SimpleTagResourcesMPE(
        num_resources=4, placement=placement,
        collect_radius=0.15, collect_reward=COLLECT_REWARD,
        circle_radius=0.6, corner_offset=0.8,
        obstacle_positions=OBSTACLES,
        pred_max_speed=PRED_MAX_SPEED, pred_accel=PRED_ACCEL,
    )


# ------------------------------------------------------------------ #
# Rollout
# ------------------------------------------------------------------ #

def rollout_one_checkpoint(algorithm, placement, seed_idx, num_eps,
                           eval_placement, rng_key):
    """Roll out `num_eps` parallel greedy episodes of length NUM_STEPS.

    `placement`      -- training placement (determines which checkpoint to load)
    `eval_placement` -- env placement used for the actual rollouts
    """
    base_env = _make_env(eval_placement)
    teams    = split_teams(base_env)
    env      = MPELogWrapper(base_env)
    wrapped  = CTRolloutManager(env, batch_size=num_eps)

    prey_name  = teams["prey"][0]
    pred_names = list(teams["pred"])
    prey_idx   = base_env.agents.index(prey_name)
    pred_idxs  = [base_env.agents.index(n) for n in pred_names]
    action_dim = wrapped.max_action_space

    # Max raw obs dim (without CTRolloutManager one-hot).
    # MAPPO was trained on pad_obs (no one-hot), so we strip the trailing
    # agent-id dims when feeding MAPPO.  IQL / OA-IQL were trained WITH
    # the CTRolloutManager one-hot, so they get the full obs.
    obs_sizes = {a: base_env.observation_space(a).shape[0]
                 for a in base_env.agents}
    max_obs = max(obs_sizes.values())          # 26 for resource env

    # --- build networks per team ---
    if algorithm == "iql":
        nets = {t: MLPQNetwork(action_dim=action_dim, hidden_dim=HIDDEN)
                for t in teams}
    elif algorithm == "oa_iql":
        nets = {
            "pred": MLPQOppNetwork(action_dim=action_dim, hidden_dim=HIDDEN,
                                   opp_n_agents=len(teams["prey"])),
            "prey": MLPQOppNetwork(action_dim=action_dim, hidden_dim=HIDDEN,
                                   opp_n_agents=len(teams["pred"])),
        }
    else:
        nets = {t: ActorLogits(action_dim=action_dim, hidden_dim=HIDDEN)
                for t in teams}

    # --- load pretrained params ---
    params = {t: load_params(_params_path(algorithm, placement, t, seed_idx))
              for t in teams}

    # --- reset ---
    rng_key, k_reset = jax.random.split(rng_key)
    obs, env_state = wrapped.batch_reset(k_reset)
    init_full = np.asarray(env_state.env_state.p_pos)       # (N, n_ent, 2)

    # --- trajectory storage ---
    prey_pos = [np.asarray(env_state.env_state.p_pos[:, prey_idx])]
    pred_pos = [np.asarray(env_state.env_state.p_pos[:, pred_idxs])]
    res_pos  = [np.asarray(env_state.env_state.resource_pos)]
    coll     = [np.asarray(env_state.env_state.collected)]
    prey_acts, prey_rews = [], []
    pred_acts, pred_rews = [], []

    for _ in range(NUM_STEPS):
        rng_key, k = jax.random.split(rng_key)
        valid = wrapped.get_valid_actions(env_state.env_state)

        all_actions = {}
        for t in teams:
            for a in teams[t]:
                if algorithm == "mappo":
                    # Strip CTRolloutManager one-hot (last N_agents dims)
                    logits = nets[t].apply(params[t], obs[a][:, :max_obs])
                    all_actions[a] = jnp.argmax(logits, axis=-1).astype(jnp.int32)
                elif algorithm == "oa_iql":
                    q, _ = nets[t].apply(params[t], obs[a])
                    q = q - (1 - valid[a]) * 1e10
                    all_actions[a] = jnp.argmax(q, axis=-1).astype(jnp.int32)
                else:  # iql
                    q = nets[t].apply(params[t], obs[a])
                    q = q - (1 - valid[a]) * 1e10
                    all_actions[a] = jnp.argmax(q, axis=-1).astype(jnp.int32)

        obs, env_state, rewards, dones, _ = wrapped.batch_step(
            k, env_state, all_actions)

        prey_pos.append(np.asarray(env_state.env_state.p_pos[:, prey_idx]))
        pred_pos.append(np.asarray(env_state.env_state.p_pos[:, pred_idxs]))
        res_pos.append(np.asarray(env_state.env_state.resource_pos))
        coll.append(np.asarray(env_state.env_state.collected))
        prey_acts.append(np.asarray(all_actions[prey_name]))
        prey_rews.append(np.asarray(rewards[prey_name]))
        pred_acts.append(np.stack(
            [np.asarray(all_actions[n]) for n in pred_names], axis=-1))
        pred_rews.append(np.stack(
            [np.asarray(rewards[n]) for n in pred_names], axis=-1))

    return dict(
        positions      = np.stack(prey_pos, axis=1).astype(np.float32),
        pred_positions = np.stack(pred_pos, axis=1).astype(np.float32),
        resource_pos   = np.stack(res_pos, axis=1).astype(np.float32),
        collected      = np.stack(coll, axis=1),
        init_full      = init_full.astype(np.float32),
        actions        = np.stack(prey_acts, axis=1).astype(np.int32),
        rewards        = np.stack(prey_rews, axis=1).astype(np.float32),
        pred_actions   = np.stack(pred_acts, axis=1).astype(np.int32),
        pred_rewards   = np.stack(pred_rews, axis=1).astype(np.float32),
    )


# ------------------------------------------------------------------ #
# Dataset builders
# ------------------------------------------------------------------ #

def build_dataset_A(algorithm, eval_placement, rng):
    parts, labels, checkpoints = [], [], []
    label_id = 0
    total = len(PLACEMENTS) * len(SEEDS_A)
    for placement in PLACEMENTS:
        for seed in SEEDS_A:
            rng, k = jax.random.split(rng)
            tag = f"{placement}_seed{seed}"
            print(f"  [A] {algorithm} {tag}  ({label_id+1}/{total})  "
                  f"N={NUM_EPS_A} ...", flush=True)
            d = rollout_one_checkpoint(algorithm, placement, seed,
                                       NUM_EPS_A, eval_placement, k)
            parts.append(d)
            labels.append(np.full(d["positions"].shape[0], label_id, np.int32))
            checkpoints.append(tag)
            label_id += 1

    out = {key: np.concatenate([p[key] for p in parts], axis=0)
           for key in parts[0]}
    out["labels"]      = np.concatenate(labels, axis=0)
    out["checkpoints"] = np.array(checkpoints)
    return out


def build_dataset_B(algorithm, b_placement, b_seed, eval_placement, rng):
    rng, k = jax.random.split(rng)
    tag = f"{b_placement}_seed{b_seed}"
    print(f"  [B] {algorithm} {tag}  N={NUM_EPS_B} ...", flush=True)
    d = rollout_one_checkpoint(algorithm, b_placement, b_seed,
                               NUM_EPS_B, eval_placement, k)
    d["labels"]      = np.full(d["positions"].shape[0], -1, np.int32)
    d["checkpoints"] = np.array([tag])
    return d


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser(
        description="Generate trajectory datasets from resource-env checkpoints.")
    ap.add_argument("--algorithm", choices=["iql", "oa_iql", "mappo"],
                    default="iql")
    ap.add_argument("--eval_placement", default="random",
                    help="Env placement mode for rollouts (default: random)")
    ap.add_argument("--b_placement", default="circle",
                    help="Training placement for the dataset-B checkpoint")
    ap.add_argument("--b_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(LOGDIR, exist_ok=True)
    rng = jax.random.PRNGKey(args.seed)

    suffix = f"_{args.algorithm}" if args.algorithm != "iql" else ""

    print(f"--- Algorithm: {args.algorithm}, eval env: {args.eval_placement} ---")

    # Dataset A
    N_A = len(PLACEMENTS) * len(SEEDS_A) * NUM_EPS_A
    print(f"Generating Reading A ({N_A} trajectories, T={NUM_STEPS}) ...")
    rng, kA = jax.random.split(rng)
    A = build_dataset_A(args.algorithm, args.eval_placement, kA)
    A["meta"] = np.array([
        f"T={NUM_STEPS}", f"HIDDEN={HIDDEN}",
        f"algorithm={args.algorithm}",
        f"eval_placement={args.eval_placement}",
    ])
    out_A = f"{LOGDIR}/trajectory_dataset_resources{suffix}_A.npz"
    np.savez_compressed(out_A, **A)
    print(f"  wrote {out_A}  positions.shape={A['positions'].shape}\n")

    # Dataset B
    print(f"Generating Reading B ({NUM_EPS_B} trajectories, T={NUM_STEPS}) ...")
    rng, kB = jax.random.split(rng)
    B = build_dataset_B(args.algorithm, args.b_placement, args.b_seed,
                        args.eval_placement, kB)
    B["meta"] = np.array([
        f"T={NUM_STEPS}", f"HIDDEN={HIDDEN}",
        f"algorithm={args.algorithm}",
        f"eval_placement={args.eval_placement}",
    ])
    out_B = f"{LOGDIR}/trajectory_dataset_resources{suffix}_B.npz"
    np.savez_compressed(out_B, **B)
    print(f"  wrote {out_B}  positions.shape={B['positions'].shape}")


if __name__ == "__main__":
    main()
