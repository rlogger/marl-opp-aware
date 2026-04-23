"""
Tree-expanded (rollout-based) planning for OA-IQL-MLP at eval time.

The opponent-action head trained as an auxiliary classifier on OA-IQL is also
a one-step opponent model. We use it here as the sampler in a Monte-Carlo
rollout planner: for each candidate root self-action, we roll out H steps
through the JaxMARL simulator, sampling opponent actions from the OA head
and letting teammates act greedily on the shared Q-net. We average over K
rollouts, then pick the root action with the highest estimated value.

Design notes
------------
- Per-agent *independent* planning (coordinate-descent style). Each predator
  plans its own action with teammates frozen at greedy-Q. This avoids the 5^3
  = 125 joint-action blowup on the predator side and makes the prey case
  (single agent) symmetric with the predator case.
- H=0 recovers argmax-Q (the greedy baseline).
- K is the branching factor over opponent actions. K=1 is a single random
  sample from the OA head (noisy but free); K>1 reduces variance.
- This planner is only used at EVAL. Training is unchanged (iql_teams_oa_mlp).
- Stepping uses the CTRolloutManager's `wrapped_step` for consistency with
  training-time obs preprocessing (padding + agent one-hot).

Usage
-----
    from planning_eval import make_plan_team_actions
    plan_pred = make_plan_team_actions(wrapped_env, "pred", K=5, H=3, gamma=0.9)
    pred_actions = plan_pred(params, obs_dict, env_state, rng)
"""
from typing import Dict, List

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


class MLPQOppNetwork(nn.Module):
    """Architecture duplicate of iql_teams_oa_mlp.MLPQOppNetwork (for loading)."""
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


class MLPQNetwork(nn.Module):
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
        q = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale),
                     bias_init=constant(0.0))(x)
        return q


def _split_teams(env) -> Dict[str, List[str]]:
    preds = [a for a in env.agents if a.startswith("adversary")]
    prey = [a for a in env.agents if a.startswith("agent")]
    return {"pred": preds, "prey": prey}


def make_plan_team_actions(wrapped_env, team_name: str, K: int = 5, H: int = 3,
                           gamma: float = 0.9, action_dim: int = 5,
                           hidden_dim: int = 128):
    """Factory returning a `plan_team(params, obs_dict, env_state, rng) -> action_dict`.

    `wrapped_env` is a CTRolloutManager (gives us `.wrapped_step`).
    `params` will be the trained OA-IQL-MLP params for team `team_name`.
    `env_state` is the MPELogWrapper state (the `.env_state` attr of the
    CTRolloutManager batch state, but *unbatched* when this function is vmapped).
    """
    env = wrapped_env._env  # MPELogWrapper(base)
    teams = _split_teams(wrapped_env)
    team_agents = teams[team_name]
    opp_name = {"pred": "prey", "prey": "pred"}[team_name]
    opp_agents = teams[opp_name]
    team_n = len(team_agents)
    opp_n = len(opp_agents)

    net = MLPQOppNetwork(action_dim=action_dim, hidden_dim=hidden_dim,
                         opp_n_agents=opp_n)

    def plan_team(params, obs_dict, env_state, rng):
        def plan_one_agent(agent_name, rng):

            def score_root_action(a_root, rng):
                def one_rollout(rng):
                    def step_fn(carry, t):
                        state, obs_d, discount, accum, rng = carry
                        rng, rng_opp, rng_step = jax.random.split(rng, 3)

                        # My action: root action at t==0, greedy afterwards
                        my_q, _ = net.apply(params, obs_d[agent_name][None])
                        my_a_greedy = jnp.argmax(my_q[0]).astype(jnp.int32)
                        my_a = jnp.where(t == 0, a_root, my_a_greedy)

                        # Teammates: greedy on own shared net
                        tm_actions = {}
                        for tm in team_agents:
                            if tm == agent_name:
                                tm_actions[tm] = my_a
                            else:
                                tm_q, _ = net.apply(params, obs_d[tm][None])
                                tm_actions[tm] = jnp.argmax(tm_q[0]).astype(jnp.int32)

                        # Opponents: sample from MY opp-head at MY obs
                        _, my_opp_logits = net.apply(params, obs_d[agent_name][None])
                        opp_logits_t = my_opp_logits[0]  # (opp_n, action_dim)
                        opp_rngs = jax.random.split(rng_opp, opp_n)
                        opp_actions = {
                            opp_agents[i]: jax.random.categorical(
                                opp_rngs[i], opp_logits_t[i]
                            ).astype(jnp.int32)
                            for i in range(opp_n)
                        }

                        all_actions = {**tm_actions, **opp_actions}
                        new_obs, new_state, rewards, dones, _ = wrapped_env.wrapped_step(
                            rng_step, state, all_actions
                        )
                        r = rewards[agent_name].astype(jnp.float32)
                        new_accum = accum + discount * r
                        new_discount = discount * gamma
                        return (new_state, new_obs, new_discount, new_accum, rng), None

                    init_carry = (env_state, obs_dict, jnp.float32(1.0),
                                  jnp.float32(0.0), rng)
                    (final_state, final_obs, final_disc, final_accum, _), _ = jax.lax.scan(
                        step_fn, init_carry, jnp.arange(H)
                    )
                    leaf_q, _ = net.apply(params, final_obs[agent_name][None])
                    v_leaf = jnp.max(leaf_q[0])
                    return final_accum + final_disc * v_leaf

                rngs = jax.random.split(rng, K)
                returns = jax.vmap(one_rollout)(rngs)
                return jnp.mean(returns)

            action_rngs = jax.random.split(rng, action_dim)
            scores = jax.vmap(score_root_action)(
                jnp.arange(action_dim), action_rngs
            )
            return jnp.argmax(scores).astype(jnp.int32)

        rngs = jax.random.split(rng, team_n)
        out = {}
        for i, ag in enumerate(team_agents):
            out[ag] = plan_one_agent(ag, rngs[i])
        return out

    return plan_team


def make_greedy_team_actions(wrapped_env, team_name: str, has_opp_head: bool,
                              action_dim: int = 5, hidden_dim: int = 128):
    """Greedy argmax-Q action selector, signature-compatible with `plan_team`."""
    teams = _split_teams(wrapped_env)
    team_agents = teams[team_name]
    opp_name = {"pred": "prey", "prey": "pred"}[team_name]
    opp_n = len(teams[opp_name])

    if has_opp_head:
        net = MLPQOppNetwork(action_dim=action_dim, hidden_dim=hidden_dim,
                             opp_n_agents=opp_n)
    else:
        net = MLPQNetwork(action_dim=action_dim, hidden_dim=hidden_dim)

    def greedy_team(params, obs_dict, env_state, rng):
        out = {}
        for ag in team_agents:
            if has_opp_head:
                q, _ = net.apply(params, obs_dict[ag][None])
            else:
                q = net.apply(params, obs_dict[ag][None])
            out[ag] = jnp.argmax(q[0]).astype(jnp.int32)
        return out

    return greedy_team
