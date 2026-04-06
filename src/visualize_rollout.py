"""Load trained two-team IQL params and roll out one episode; save GIF."""
import argparse
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

# local
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iql_teams import RNNQNetwork, ScannedRNN, split_teams

from jaxmarl import make
from jaxmarl.wrappers.baselines import load_params, CTRolloutManager, MPELogWrapper
from jaxmarl.environments.mpe.mpe_visualizer import MPEVisualizer


def greedy(q, valid):
    unavail = 1 - valid
    q = q - (unavail * 1e10)
    return jnp.argmax(q, axis=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_params", required=True)
    p.add_argument("--prey_params", required=True)
    p.add_argument("--env", default="MPE_simple_tag_v3")
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="plots/rollout.gif")
    args = p.parse_args()

    base_env = make(args.env)
    teams = split_teams(base_env)

    # same wrapping as training for state/obs consistency
    env = MPELogWrapper(base_env)
    wrapped = CTRolloutManager(env, batch_size=1)

    nets = {
        t: RNNQNetwork(action_dim=wrapped.max_action_space, hidden_dim=args.hidden_size)
        for t in teams
    }
    params = {
        "pred": load_params(args.pred_params),
        "prey": load_params(args.prey_params),
    }

    rng = jax.random.PRNGKey(args.seed)
    rng, key_reset = jax.random.split(rng)
    obs, env_state = wrapped.batch_reset(key_reset)

    # hidden states per team (shape: (num_agents, 1 batch, hidden))
    hs = {
        t: ScannedRNN.initialize_carry(args.hidden_size, len(teams[t]), 1)
        for t in teams
    }
    dones = {a: jnp.zeros(1, dtype=bool) for a in env.agents + ["__all__"]}

    state_seq = [env_state.env_state]
    reward_seq = []

    for step in range(args.steps):
        rng, key_step = jax.random.split(rng)
        valid = wrapped.get_valid_actions(env_state.env_state)

        actions = {}
        for t in teams:
            ags = teams[t]
            _obs = jnp.stack([obs[a] for a in ags], axis=0)[:, None]  # (n, 1t, 1b, obs)
            _dn = jnp.stack([dones[a] for a in ags], axis=0)[:, None]
            new_hs, q = jax.vmap(nets[t].apply, in_axes=(None, 0, 0, 0))(
                params[t], hs[t], _obs, _dn
            )
            q = q.squeeze(axis=1)
            va = jnp.stack([valid[a] for a in ags], axis=0)
            a_team = greedy(q, va)
            hs[t] = new_hs
            for i, name in enumerate(ags):
                actions[name] = a_team[i]

        obs, env_state, rewards, new_dones, _ = wrapped.batch_step(
            key_step, env_state, actions
        )
        dones = new_dones
        state_seq.append(env_state.env_state)
        reward_seq.append({k: float(v[0]) for k, v in rewards.items() if k != "__all__"})

    # MPEVisualizer wants per-step states (not batched). Unbatch.
    def unbatch(s):
        return jax.tree.map(lambda x: np.asarray(x[0]) if hasattr(x, 'shape') and x.ndim > 0 else x, s)

    state_seq_unb = [unbatch(s) for s in state_seq]

    print(f"Rollout finished: {len(state_seq_unb)} states")
    print("Avg predator reward per step:",
          np.mean([sum(r[a] for a in teams['pred']) for r in reward_seq]))
    print("Avg prey reward per step:",
          np.mean([r[teams['prey'][0]] for r in reward_seq]))

    viz = MPEVisualizer(base_env, state_seq_unb)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    viz.animate(save_fname=args.out, view=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
