"""
Opponent-Aware IQL (OA-IQL), MLP variant -- no RNN.

Structural twin of iql_teams_oa.py with the ScannedRNN stripped. Each team's
Q-network is an MLP with a shared trunk and two heads:
  - Q head    -> action values (action_dim per agent)
  - Opp head  -> logits over opponent team's actions, (opp_n_agents, action_dim)

Loss per team = Q-loss + OPP_AUX_COEF * CE(opp_action_pred, opp_action_true)

The opp head is used two ways:
  1. Training-time representation-learning auxiliary (this file)
  2. Eval-time one-step opponent model for tree-expanded planning
     (see src/planning_eval.py)

Run:
    python src/iql_teams_oa_mlp.py +alg=ql_teams_oa_mlp_simple_tag
"""
import os
import copy
import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, List

import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import flashbax as fbx
import wandb

from jaxmarl import make
from jaxmarl.wrappers.baselines import MPELogWrapper, CTRolloutManager

from simple_tag_static import SimpleTagStaticMPE


# ---------- Networks ----------

class MLPQOppNetwork(nn.Module):
    """Two-headed MLP Q + opponent-action prediction.

    Shared trunk: obs -> Dense -> ReLU -> Dense -> ReLU.
    Q head: Dense -> (..., action_dim)
    Opp head: Dense -> (..., opp_n_agents, action_dim)
    """
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


@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    avail_actions: dict


class CustomTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def split_teams(env) -> Dict[str, List[str]]:
    preds = [a for a in env.agents if a.startswith("adversary")]
    prey = [a for a in env.agents if a.startswith("agent")]
    assert preds and prey, f"Could not split teams from agents={env.agents}"
    return {"pred": preds, "prey": prey}


def make_train(config, env):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    teams = split_teams(env)
    team_names = list(teams.keys())
    opp_of = {"pred": "prey", "prey": "pred"}

    eps_scheduler = optax.linear_schedule(
        init_value=config["EPS_START"],
        end_value=config["EPS_FINISH"],
        transition_steps=config["EPS_DECAY"] * config["NUM_UPDATES"],
    )

    def get_greedy_actions(q_vals, valid_actions):
        unavail = 1 - valid_actions
        q_vals = q_vals - (unavail * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    def eps_greedy(rng, q_vals, eps, valid_actions):
        rng_a, rng_e = jax.random.split(rng)
        greedy = get_greedy_actions(q_vals, valid_actions)

        def _rand(rng, va):
            return jax.random.choice(rng, jnp.arange(va.shape[-1]),
                                     p=va * 1.0 / jnp.sum(va, axis=-1))
        rngs = jax.random.split(rng_a, valid_actions.shape[0])
        rand_a = jax.vmap(_rand)(rngs, valid_actions)
        return jnp.where(jax.random.uniform(rng_e, greedy.shape) < eps, rand_a, greedy)

    def batchify_team(x: dict, team_agents: List[str]):
        return jnp.stack([x[a] for a in team_agents], axis=0)

    def unbatchify_team(arr, team_agents: List[str]):
        return {a: arr[i] for i, a in enumerate(team_agents)}

    def train(rng):
        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = CTRolloutManager(env, batch_size=config["TEST_NUM_ENVS"])

        networks = {
            t: MLPQOppNetwork(
                action_dim=wrapped_env.max_action_space,
                hidden_dim=config["HIDDEN_SIZE"],
                opp_n_agents=len(teams[opp_of[t]]),
            )
            for t in team_names
        }

        def create_agent(rng, team):
            init_obs = jnp.zeros((1, wrapped_env.obs_size))
            params = networks[team].init(rng, init_obs)

            lr_sched = optax.linear_schedule(
                init_value=config["LR"], end_value=1e-10,
                transition_steps=config["NUM_EPOCHS"] * config["NUM_UPDATES"],
            )
            lr = lr_sched if config.get("LR_LINEAR_DECAY", False) else config["LR"]

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )
            return CustomTrainState.create(
                apply_fn=networks[team].apply,
                params=params,
                target_network_params=params,
                tx=tx,
            )

        rng, rng_pred, rng_prey = jax.random.split(rng, 3)
        train_states = {
            "pred": create_agent(rng_pred, "pred"),
            "prey": create_agent(rng_prey, "prey"),
        }

        # INIT BUFFER
        def _env_sample_step(env_state, _):
            rng_d, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3)
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {
                a: wrapped_env.batch_sample(key_a[i], a)
                for i, a in enumerate(env.agents)
            }
            avail = wrapped_env.get_valid_actions(env_state.env_state)
            obs, env_state, rewards, dones, _ = wrapped_env.batch_step(
                key_s, env_state, actions
            )
            rewards = jax.tree.map(lambda r: r.astype(jnp.float32), rewards)
            ts = Timestep(obs=obs, actions=actions, rewards=rewards,
                          dones=dones, avail_actions=avail)
            return env_state, ts

        _, _env_state = wrapped_env.batch_reset(rng)
        _, sample_traj = jax.lax.scan(_env_sample_step, _env_state, None, config["NUM_STEPS"])
        sample_traj_unbatched = jax.tree.map(lambda x: x[:, 0], sample_traj)
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched)

        def _update_step(runner_state, _):
            train_states, buffer_state, test_state, rng = runner_state

            def _step_env(carry, _):
                last_obs, last_dones, env_state, rng = carry
                rng, rng_a_pred, rng_a_prey, rng_s = jax.random.split(rng, 4)
                avail = wrapped_env.get_valid_actions(env_state.env_state)
                eps = eps_scheduler(train_states["pred"].n_updates)

                def team_act(team, rng_a):
                    ags = teams[team]
                    _obs = batchify_team(last_obs, ags)
                    q, _opp = jax.vmap(networks[team].apply, in_axes=(None, 0))(
                        train_states[team].params, _obs
                    )
                    va = batchify_team(avail, ags)
                    rngs = jax.random.split(rng_a, len(ags))
                    acts = jax.vmap(eps_greedy, in_axes=(0, 0, None, 0))(
                        rngs, q, eps, va
                    )
                    return unbatchify_team(acts, ags)

                acts_pred = team_act("pred", rng_a_pred)
                acts_prey = team_act("prey", rng_a_prey)
                actions = {**acts_pred, **acts_prey}

                new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, actions
                )
                ts = Timestep(
                    obs=last_obs, actions=actions,
                    rewards=jax.tree.map(
                        lambda x: (config.get("REW_SCALE", 1) * x).astype(jnp.float32),
                        rewards,
                    ),
                    dones=last_dones, avail_actions=avail,
                )
                return (new_obs, dones, new_env_state, rng), (ts, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {a: jnp.zeros(config["NUM_ENVS"], dtype=bool)
                          for a in env.agents + ["__all__"]}
            rng, _rng = jax.random.split(rng)
            _, (timesteps, infos) = jax.lax.scan(
                _step_env,
                (init_obs, init_dones, env_state, _rng),
                None, config["NUM_STEPS"],
            )

            train_states = {
                t: ts.replace(timesteps=ts.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"])
                for t, ts in train_states.items()
            }

            buffer_traj_batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis], timesteps
            )
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            def _learn_team(team, train_state, minibatch):
                ags = teams[team]
                opp_ags = teams[opp_of[team]]
                _obs = batchify_team(minibatch.obs, ags)
                _dones = batchify_team(minibatch.dones, ags)
                _actions = batchify_team(minibatch.actions, ags)
                _rewards = batchify_team(minibatch.rewards, ags)
                _avail = batchify_team(minibatch.avail_actions, ags)
                _opp_actions = batchify_team(minibatch.actions, opp_ags)

                q_next_target, _ = jax.vmap(
                    networks[team].apply, in_axes=(None, 0)
                )(train_state.target_network_params, _obs)

                def _loss_fn(params):
                    q, opp_logits = jax.vmap(
                        networks[team].apply, in_axes=(None, 0)
                    )(params, _obs)
                    chosen = jnp.take_along_axis(
                        q, _actions[..., np.newaxis], axis=-1
                    ).squeeze(-1)
                    unavail = 1 - _avail
                    valid_q = q - (unavail * 1e10)
                    q_next = jnp.take_along_axis(
                        q_next_target,
                        jnp.argmax(valid_q, axis=-1)[..., np.newaxis],
                        axis=-1,
                    ).squeeze(-1)
                    target = (_rewards[:, :-1]
                              + (1 - _dones[:, :-1]) * config["GAMMA"] * q_next[:, 1:])
                    chosen = chosen[:, :-1]
                    q_loss = jnp.mean((chosen - jax.lax.stop_gradient(target)) ** 2)

                    # Opponent-action CE loss
                    # opp_logits: (team_n, B, T, opp_n, action_dim)
                    # _opp_actions: (opp_n, B, T)
                    opp_tgt = jnp.broadcast_to(
                        _opp_actions[None],
                        (opp_logits.shape[0],) + _opp_actions.shape,
                    )  # (team_n, opp_n, B, T)
                    opp_tgt = jnp.transpose(opp_tgt, (0, 2, 3, 1))  # (team_n, B, T, opp_n)
                    logp = jax.nn.log_softmax(opp_logits, axis=-1)
                    ce = -jnp.take_along_axis(
                        logp, opp_tgt[..., None], axis=-1
                    ).squeeze(-1)
                    opp_ce = jnp.mean(ce)
                    opp_pred = jnp.argmax(opp_logits, axis=-1)
                    opp_acc = jnp.mean((opp_pred == opp_tgt).astype(jnp.float32))

                    total = q_loss + config.get("OPP_AUX_COEF", 0.5) * opp_ce
                    return total, (q_loss, chosen.mean(), opp_ce, opp_acc)

                (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
                q_loss, qv, opp_ce, opp_acc = aux
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(grad_steps=train_state.grad_steps + 1)
                return train_state, (q_loss, qv, opp_ce, opp_acc)

            def _learn_phase(carry, _):
                train_states, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                minibatch = jax.tree.map(
                    lambda x: jnp.swapaxes(x[:, 0], 0, 1), minibatch
                )
                new_states = {}
                losses = {}
                for t in team_names:
                    ns, (q_loss, qv, opp_ce, opp_acc) = _learn_team(
                        t, train_states[t], minibatch
                    )
                    new_states[t] = ns
                    losses[t] = (q_loss, qv, opp_ce, opp_acc)
                return (new_states, rng), losses

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                buffer.can_sample(buffer_state)
                & (train_states["pred"].timesteps > config["LEARNING_STARTS"])
            )
            (train_states, rng), losses = jax.lax.cond(
                is_learn_time,
                lambda ts, r: jax.lax.scan(_learn_phase, (ts, r), None, config["NUM_EPOCHS"]),
                lambda ts, r: (
                    (ts, r),
                    {t: (jnp.zeros(config["NUM_EPOCHS"]),
                         jnp.zeros(config["NUM_EPOCHS"]),
                         jnp.zeros(config["NUM_EPOCHS"]),
                         jnp.zeros(config["NUM_EPOCHS"]))
                     for t in team_names},
                ),
                train_states, _rng,
            )

            def _maybe_target(ts):
                return jax.lax.cond(
                    ts.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                    lambda s: s.replace(
                        target_network_params=optax.incremental_update(
                            s.params, s.target_network_params, config["TAU"]
                        )
                    ),
                    lambda s: s,
                    operand=ts,
                )
            train_states = {t: _maybe_target(ts) for t, ts in train_states.items()}
            train_states = {t: ts.replace(n_updates=ts.n_updates + 1)
                            for t, ts in train_states.items()}

            metrics = {
                "env_step": train_states["pred"].timesteps,
                "update_steps": train_states["pred"].n_updates,
            }
            for t in team_names:
                metrics[f"{t}/loss"] = losses[t][0].mean()
                metrics[f"{t}/qvals"] = losses[t][1].mean()
                metrics[f"{t}/opp_ce"] = losses[t][2].mean()
                metrics[f"{t}/opp_acc"] = losses[t][3].mean()

            flat_infos = jax.tree.map(lambda x: x.mean(), infos)
            metrics.update({f"all/{k}": v for k, v in flat_infos.items()})
            for t in team_names:
                idxs = jnp.array([env.agents.index(a) for a in teams[t]])
                tm = jax.tree.map(
                    lambda x: x[..., idxs].mean(), infos
                )
                metrics.update({f"{t}/{k}": v for k, v in tm.items()})

            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    train_states["pred"].n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"]) == 0,
                    lambda _: get_greedy_metrics(_rng, train_states),
                    lambda _: test_state,
                    operand=None,
                )
                metrics.update({f"test/{k}": v for k, v in test_state.items()})

            if config["WANDB_MODE"] != "disabled":
                def cb(m, seed):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        m.update({f"rng{int(seed)}/{k}": v for k, v in m.items()})
                    wandb.log(m)
                jax.debug.callback(cb, metrics, original_seed)

            return (train_states, buffer_state, test_state, rng), metrics

        def get_greedy_metrics(rng, train_states):
            if not config.get("TEST_DURING_TRAINING", True):
                return None

            def _greedy_step(step_state, _):
                (params_pred, params_prey, env_state,
                 last_obs, last_dones, rng) = step_state
                rng, key_s = jax.random.split(rng)
                valid = test_env.get_valid_actions(env_state.env_state)

                def fwd(team, params):
                    ags = teams[team]
                    _obs = batchify_team(last_obs, ags)
                    q, _ = jax.vmap(networks[team].apply, in_axes=(None, 0))(params, _obs)
                    va = batchify_team(valid, ags)
                    a = get_greedy_actions(q, va)
                    return unbatchify_team(a, ags)

                acts_pred = fwd("pred", params_pred)
                acts_prey = fwd("prey", params_prey)
                actions = {**acts_pred, **acts_prey}
                obs, env_state, rewards, dones, infos = test_env.batch_step(
                    key_s, env_state, actions
                )
                return (params_pred, params_prey, env_state, obs, dones, rng), (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {a: jnp.zeros(config["TEST_NUM_ENVS"], dtype=bool)
                          for a in env.agents + ["__all__"]}
            rng, _rng = jax.random.split(rng)
            step_state = (train_states["pred"].params, train_states["prey"].params,
                          env_state, init_obs, init_dones, _rng)
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_step, step_state, None, config["TEST_NUM_STEPS"]
            )

            metrics = {}
            for t in team_names:
                idxs = jnp.array([env.agents.index(a) for a in teams[t]])
                m = jax.tree.map(
                    lambda x: jnp.nanmean(
                        jnp.where(infos["returned_episode"][..., idxs], x[..., idxs], jnp.nan)
                    ),
                    infos,
                )
                metrics.update({f"{t}/{k}": v for k, v in m.items()})
            return metrics

        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, train_states)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_states, buffer_state, test_state, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def env_from_config(config):
    if config.get("USE_STATIC_OBSTACLES", False):
        env = SimpleTagStaticMPE(
            obstacle_positions=config.get("OBSTACLE_POSITIONS", None),
            shape_coef=float(config.get("SHAPE_COEF", 0.0)),
            shape_direction=config.get("SHAPE_DIRECTION", "ccw"),
            shape_r_in=float(config.get("SHAPE_R_IN", 0.15)),
            shape_r_out=float(config.get("SHAPE_R_OUT", 0.6)),
            **config["ENV_KWARGS"],
        )
    else:
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = MPELogWrapper(env)
    return env, config["ENV_NAME"]


def single_run(config):
    config = {**config, **config["alg"]}
    print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = config.get("ALG_NAME", "iql_teams_oa_mlp")
    env, env_name = env_from_config(copy.deepcopy(config))

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[alg_name.upper(), env_name.upper(), f"jax_{jax.__version__}"],
        name=f"{alg_name}_{env_name}",
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    t0 = time.time()
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))
    dt = time.time() - t0
    print(f"Training done in {dt:.1f}s")

    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params
        states = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'),
        )
        for t, ts in states.items():
            for i in range(config["NUM_SEEDS"]):
                params = jax.tree.map(lambda x: x[i], ts.params)
                path = os.path.join(
                    save_dir,
                    f'{alg_name}_{env_name}_{t}_seed{config["SEED"]}_vmap{i}.safetensors',
                )
                save_params(params, path)
        metrics = outs["metrics"]
        metrics_np = jax.tree.map(lambda x: np.asarray(x), metrics)
        np.savez_compressed(
            os.path.join(save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_metrics.npz'),
            **{k.replace("/", "__"): v for k, v in metrics_np.items()
               if isinstance(v, np.ndarray)},
        )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    single_run(config)


if __name__ == "__main__":
    main()
