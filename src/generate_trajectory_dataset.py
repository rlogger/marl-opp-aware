"""Generate prey-only trajectory datasets from trained checkpoints.

Two datasets are produced in one run:

  Reading A — multi-checkpoint dataset.
    9 Exp 2 prey checkpoints {static_baseline, static_cw, static_ccw} x {seed 0,1,2}.
    NUM_EPS_A rollouts per checkpoint, T = NUM_STEPS steps each.
    Each trajectory carries a ground-truth label (the checkpoint id).
    Use this to verify that distinct shaping policies produce distinguishable
    trajectory distributions (a clustering algorithm should recover the labels).

  Reading B — single-checkpoint dataset.
    One prey checkpoint (default static_baseline seed 0). NUM_EPS_B rollouts.
    No labels. Use this to test whether a single policy's rollout distribution
    is itself multi-modal (e.g., one policy producing several recurring
    behaviour patterns under varying initial conditions).

Outputs (both .npz under logs/MPE_simple_tag_v3/):
  trajectory_dataset_A.npz   — multi-checkpoint, labelled
  trajectory_dataset_B.npz   — single-checkpoint, unlabelled

Schema (per npz):
  positions  (N, T+1, 2)   float32   prey (x,y) per step (the only "trajectory")
  init_full  (N, n_ent, 2) float32   reset positions for all entities (context)
  actions    (N, T)        int32     prey discrete action per step
  rewards    (N, T)        float32   prey reward per step (excludes shaping bonus)
  labels     (N,)          int32     checkpoint id (Reading A only; -1 for B)
  checkpoints (K,)         <U64      checkpoint name lookup
  meta       dict-like                hyperparameters
"""
import os
import sys
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iql_teams import RNNQNetwork, ScannedRNN, split_teams  # Exp 2 used the RNN backbone
from simple_tag_static import SimpleTagStaticMPE
from jaxmarl.wrappers.baselines import CTRolloutManager, MPELogWrapper, load_params


LOGDIR     = "logs/MPE_simple_tag_v3"
NUM_STEPS  = 20         # T (trajectory horizon)
HIDDEN     = 64         # Exp 2 trained with HIDDEN_SIZE=64
OBSTACLES  = [[0.5, 0.5], [-0.5, -0.5]]

# Reading A: 9 checkpoints x NUM_EPS_A
CHECKPOINTS_A = [
    ("static_baseline", 0), ("static_baseline", 1), ("static_baseline", 2),
    ("static_cw",       0), ("static_cw",       1), ("static_cw",       2),
    ("static_ccw",      0), ("static_ccw",      1), ("static_ccw",      2),
]
NUM_EPS_A = 100   # per checkpoint -> 900 trajectories total

# Reading B: 1 checkpoint x NUM_EPS_B (same N as A so dataset sizes match)
CHECKPOINT_B = ("static_baseline", 0)
NUM_EPS_B    = 900


# --------------------------------------------------------------------------- #
def _variant_kwargs(variant: str) -> dict:
    if variant == "static_baseline":
        return dict(shape_coef=0.0, shape_direction="ccw")
    if variant == "static_cw":
        return dict(shape_coef=0.1, shape_direction="cw")
    if variant == "static_ccw":
        return dict(shape_coef=0.1, shape_direction="ccw")
    raise ValueError(variant)


def _params_path(variant: str, team: str, seed_idx: int) -> str:
    return (
        f"{LOGDIR}/iql_teams_{variant}_MPE_simple_tag_v3_"
        f"{team}_seed0_vmap{seed_idx}.safetensors"
    )


def rollout_one_checkpoint(
    variant: str,
    seed_idx: int,
    num_eps: int,
    rng_key: jax.random.PRNGKey,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out `num_eps` parallel greedy episodes of length NUM_STEPS.

    Returns:
        prey_pos   (N, T+1, 2)
        init_full  (N, n_entities, 2)
        prey_act   (N, T)
        prey_rew   (N, T)
    """
    cfg = _variant_kwargs(variant)
    base_env = SimpleTagStaticMPE(obstacle_positions=OBSTACLES, **cfg)
    teams    = split_teams(base_env)
    env      = MPELogWrapper(base_env)
    wrapped  = CTRolloutManager(env, batch_size=num_eps)

    prey_name = teams["prey"][0]
    prey_idx  = base_env.agents.index(prey_name)

    nets = {t: RNNQNetwork(action_dim=wrapped.max_action_space, hidden_dim=HIDDEN)
            for t in teams}
    params = {
        "pred": load_params(_params_path(variant, "pred", seed_idx)),
        "prey": load_params(_params_path(variant, "prey", seed_idx)),
    }

    rng_key, k_reset = jax.random.split(rng_key)
    obs, env_state = wrapped.batch_reset(k_reset)
    init_full      = np.asarray(env_state.env_state.p_pos)        # (N, n_ent, 2)

    hs = {t: ScannedRNN.initialize_carry(HIDDEN, len(teams[t]), num_eps) for t in teams}
    dones = {a: jnp.zeros(num_eps, dtype=bool) for a in env.agents + ["__all__"]}

    prey_pos_per_step = [np.asarray(env_state.env_state.p_pos[:, prey_idx])]   # (N, 2)
    prey_act_per_step = []
    prey_rew_per_step = []

    for _ in range(NUM_STEPS):
        rng_key, k = jax.random.split(rng_key)
        valid = wrapped.get_valid_actions(env_state.env_state)

        all_actions = {}
        for t in teams:
            ags = teams[t]
            _obs = jnp.stack([obs[a] for a in ags], axis=0)[:, None]   # (n_ag,1,N,obs)
            _dn  = jnp.stack([dones[a] for a in ags], axis=0)[:, None]
            new_hs, q = jax.vmap(nets[t].apply, in_axes=(None, 0, 0, 0))(
                params[t], hs[t], _obs, _dn
            )
            q = q.squeeze(axis=1)                                       # (n_ag,N,A)
            va = jnp.stack([valid[a] for a in ags], axis=0)
            q  = q - (1 - va) * 1e10
            a_team = jnp.argmax(q, axis=-1)                              # (n_ag,N)
            hs[t] = new_hs
            for i, name in enumerate(ags):
                all_actions[name] = a_team[i]

        obs, env_state, rewards, dones, _ = wrapped.batch_step(k, env_state, all_actions)

        prey_pos_per_step.append(np.asarray(env_state.env_state.p_pos[:, prey_idx]))
        prey_act_per_step.append(np.asarray(all_actions[prey_name]))
        prey_rew_per_step.append(np.asarray(rewards[prey_name]))

    prey_pos = np.stack(prey_pos_per_step, axis=1)     # (N, T+1, 2)
    prey_act = np.stack(prey_act_per_step, axis=1)     # (N, T)
    prey_rew = np.stack(prey_rew_per_step, axis=1)     # (N, T)
    return prey_pos.astype(np.float32), init_full.astype(np.float32), \
           prey_act.astype(np.int32), prey_rew.astype(np.float32)


# --------------------------------------------------------------------------- #
def build_dataset_A(rng: jax.random.PRNGKey) -> dict:
    pos_all, init_all, act_all, rew_all, lab_all = [], [], [], [], []
    for label_id, (variant, seed) in enumerate(CHECKPOINTS_A):
        rng, k = jax.random.split(rng)
        print(f"  [A] {variant} seed={seed}  ({label_id+1}/{len(CHECKPOINTS_A)})  "
              f"N={NUM_EPS_A} ...", flush=True)
        pos, init, act, rew = rollout_one_checkpoint(variant, seed, NUM_EPS_A, k)
        pos_all.append(pos); init_all.append(init); act_all.append(act); rew_all.append(rew)
        lab_all.append(np.full(pos.shape[0], label_id, dtype=np.int32))
    return dict(
        positions   = np.concatenate(pos_all,  axis=0),
        init_full   = np.concatenate(init_all, axis=0),
        actions     = np.concatenate(act_all,  axis=0),
        rewards     = np.concatenate(rew_all,  axis=0),
        labels      = np.concatenate(lab_all,  axis=0),
        checkpoints = np.array([f"{v}_seed{s}" for v, s in CHECKPOINTS_A]),
    )


def build_dataset_B(rng: jax.random.PRNGKey) -> dict:
    variant, seed = CHECKPOINT_B
    rng, k = jax.random.split(rng)
    print(f"  [B] {variant} seed={seed}  N={NUM_EPS_B} ...", flush=True)
    pos, init, act, rew = rollout_one_checkpoint(variant, seed, NUM_EPS_B, k)
    return dict(
        positions   = pos,
        init_full   = init,
        actions     = act,
        rewards     = rew,
        labels      = np.full(pos.shape[0], -1, dtype=np.int32),
        checkpoints = np.array([f"{variant}_seed{seed}"]),
    )


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    rng = jax.random.PRNGKey(0)

    print(f"Generating Reading A (multi-checkpoint, N = {len(CHECKPOINTS_A) * NUM_EPS_A}, T = {NUM_STEPS}) ...")
    rng, kA = jax.random.split(rng)
    A = build_dataset_A(kA)
    A["meta"] = np.array([f"T={NUM_STEPS}", f"HIDDEN={HIDDEN}", "prey_only=True", "obstacles=(0.5,0.5),(-0.5,-0.5)"])
    out_A = f"{LOGDIR}/trajectory_dataset_A.npz"
    np.savez_compressed(out_A, **A)
    print(f"  wrote {out_A}  positions.shape={A['positions'].shape}\n")

    print(f"Generating Reading B (single-checkpoint, N = {NUM_EPS_B}, T = {NUM_STEPS}) ...")
    rng, kB = jax.random.split(rng)
    B = build_dataset_B(kB)
    B["meta"] = np.array([f"T={NUM_STEPS}", f"HIDDEN={HIDDEN}", "prey_only=True"])
    out_B = f"{LOGDIR}/trajectory_dataset_B.npz"
    np.savez_compressed(out_B, **B)
    print(f"  wrote {out_B}  positions.shape={B['positions'].shape}")


if __name__ == "__main__":
    main()
