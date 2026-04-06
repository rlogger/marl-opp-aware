"""
Two-team independent IQL for MPE simple_tag (Predator-Prey).

Baseline for the model-based co-training project. Unlike the stock
JaxMARL iql_rnn.py -- which shares a single Q-network across *all* agents --
this script trains TWO separate IQL learners simultaneously:

  - Predator team ("adversary_*"): shared params within the team, cooperative.
  - Prey team ("agent_*"): own network, own buffer, own target.

Both teams learn against each other (co-training). Returns are logged per team
so you can see predator capture-rate up and prey escape-return up over training.

Run:
    python src/iql_teams.py +alg=ql_teams_simple_tag
"""
import os
import copy
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
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

class ScannedRNN(nn.Module):
    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class RNNQNetwork(nn.Module):
    action_dim: int
    hidden_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, hidden, obs, dones):
        emb = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale),
                       bias_init=constant(0.0))(obs)
        emb = nn.relu(emb)
        hidden, emb = ScannedRNN()(hidden, (emb, dones))
        q = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale),
                     bias_init=constant(0.0))(emb)
        return hidden, q


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


# ---------- Team split ----------

def split_teams(env) -> Dict[str, List[str]]:
    """For MPE simple_tag: adversaries = predators, good_agents = prey."""
    preds = [a for a in env.agents if a.startswith("adversary")]
    prey = [a for a in env.agents if a.startswith("agent")]
    assert preds and prey, f"Could not split teams from agents={env.agents}"
    return {"pred": preds, "prey": prey}


# ---------- Training ----------

def make_train(config, env):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    teams = split_teams(env)
    team_names = list(teams.keys())  # ["pred", "prey"]

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

        # one network per team (shared obs_size/action_dim after padding)
        networks = {
            t: RNNQNetwork(
                action_dim=wrapped_env.max_action_space,
                hidden_dim=config["HIDDEN_SIZE"],
            )
            for t in team_names
        }

        def create_agent(rng, team):
            init_x = (
                jnp.zeros((1, 1, wrapped_env.obs_size)),
                jnp.zeros((1, 1)),
            )
            init_hs = ScannedRNN.initialize_carry(config["HIDDEN_SIZE"], 1)
            params = networks[team].init(rng, init_hs, *init_x)

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

        # TRAIN LOOP
        def _update_step(runner_state, _):
            train_states, buffer_state, test_state, rng = runner_state

            # SAMPLE
            def _step_env(carry, _):
                hs_pred, hs_prey, last_obs, last_dones, env_state, rng = carry
                rng, rng_a_pred, rng_a_prey, rng_s = jax.random.split(rng, 4)

                avail = wrapped_env.get_valid_actions(env_state.env_state)
                eps = eps_scheduler(train_states["pred"].n_updates)

                # per-team forward
                def team_act(team, rng_a, hs):
                    ags = teams[team]
                    _obs = batchify_team(last_obs, ags)[:, np.newaxis]
                    _dn = batchify_team(last_dones, ags)[:, np.newaxis]
                    new_hs, q = jax.vmap(
                        networks[team].apply, in_axes=(None, 0, 0, 0)
                    )(train_states[team].params, hs, _obs, _dn)
                    q = q.squeeze(axis=1)
                    va = batchify_team(avail, ags)
                    rngs = jax.random.split(rng_a, len(ags))
                    acts = jax.vmap(eps_greedy, in_axes=(0, 0, None, 0))(
                        rngs, q, eps, va
                    )
                    return new_hs, unbatchify_team(acts, ags)

                new_hs_pred, acts_pred = team_act("pred", rng_a_pred, hs_pred)
                new_hs_prey, acts_prey = team_act("prey", rng_a_prey, hs_prey)
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
                return (new_hs_pred, new_hs_prey, new_obs, dones,
                        new_env_state, rng), (ts, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {a: jnp.zeros(config["NUM_ENVS"], dtype=bool)
                          for a in env.agents + ["__all__"]}
            init_hs_pred = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(teams["pred"]), config["NUM_ENVS"]
            )
            init_hs_prey = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(teams["prey"]), config["NUM_ENVS"]
            )
            rng, _rng = jax.random.split(rng)
            _, (timesteps, infos) = jax.lax.scan(
                _step_env,
                (init_hs_pred, init_hs_prey, init_obs, init_dones, env_state, _rng),
                None, config["NUM_STEPS"],
            )

            train_states = {
                t: ts.replace(timesteps=ts.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"])
                for t, ts in train_states.items()
            }

            # BUFFER ADD
            buffer_traj_batch = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1)[:, np.newaxis], timesteps
            )
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # LEARN per team
            def _learn_team(team, train_state, minibatch):
                ags = teams[team]
                init_hs = ScannedRNN.initialize_carry(
                    config["HIDDEN_SIZE"], len(ags), config["BUFFER_BATCH_SIZE"]
                )
                _obs = batchify_team(minibatch.obs, ags)
                _dones = batchify_team(minibatch.dones, ags)
                _actions = batchify_team(minibatch.actions, ags)
                _rewards = batchify_team(minibatch.rewards, ags)
                _avail = batchify_team(minibatch.avail_actions, ags)

                _, q_next_target = jax.vmap(
                    networks[team].apply, in_axes=(None, 0, 0, 0)
                )(train_state.target_network_params, init_hs, _obs, _dones)

                def _loss_fn(params):
                    _, q = jax.vmap(networks[team].apply, in_axes=(None, 0, 0, 0))(
                        params, init_hs, _obs, _dones
                    )
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
                    loss = jnp.mean((chosen - jax.lax.stop_gradient(target)) ** 2)
                    return loss, chosen.mean()

                (loss, qv), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(grad_steps=train_state.grad_steps + 1)
                return train_state, (loss, qv)

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
                    ns, (loss, qv) = _learn_team(t, train_states[t], minibatch)
                    new_states[t] = ns
                    losses[t] = (loss, qv)
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
                    {t: (jnp.zeros(config["NUM_EPOCHS"]), jnp.zeros(config["NUM_EPOCHS"]))
                     for t in team_names},
                ),
                train_states, _rng,
            )

            # TARGET UPDATE
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

            # METRICS
            metrics = {
                "env_step": train_states["pred"].timesteps,
                "update_steps": train_states["pred"].n_updates,
            }
            for t in team_names:
                metrics[f"{t}/loss"] = losses[t][0].mean()
                metrics[f"{t}/qvals"] = losses[t][1].mean()

            # per-agent rollout infos (team-mean of returned_episode_returns etc.)
            flat_infos = jax.tree.map(lambda x: x.mean(), infos)
            metrics.update({f"all/{k}": v for k, v in flat_infos.items()})
            for t in team_names:
                idxs = jnp.array([env.agents.index(a) for a in teams[t]])
                tm = jax.tree.map(
                    lambda x: x[..., idxs].mean(), infos
                )
                metrics.update({f"{t}/{k}": v for k, v in tm.items()})

            # EVAL
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
                 last_obs, last_dones, hs_pred, hs_prey, rng) = step_state
                rng, key_s = jax.random.split(rng)
                valid = test_env.get_valid_actions(env_state.env_state)

                def fwd(team, params, hs):
                    ags = teams[team]
                    _obs = batchify_team(last_obs, ags)[:, np.newaxis]
                    _dn = batchify_team(last_dones, ags)[:, np.newaxis]
                    new_hs, q = jax.vmap(networks[team].apply, in_axes=(None, 0, 0, 0))(
                        params, hs, _obs, _dn
                    )
                    q = q.squeeze(axis=1)
                    va = batchify_team(valid, ags)
                    a = get_greedy_actions(q, va)
                    return new_hs, unbatchify_team(a, ags)

                new_hs_pred, acts_pred = fwd("pred", params_pred, hs_pred)
                new_hs_prey, acts_prey = fwd("prey", params_prey, hs_prey)
                actions = {**acts_pred, **acts_prey}
                obs, env_state, rewards, dones, infos = test_env.batch_step(
                    key_s, env_state, actions
                )
                return (params_pred, params_prey, env_state, obs, dones,
                        new_hs_pred, new_hs_prey, rng), (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {a: jnp.zeros(config["TEST_NUM_ENVS"], dtype=bool)
                          for a in env.agents + ["__all__"]}
            rng, _rng = jax.random.split(rng)
            hs_pred = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(teams["pred"]), config["TEST_NUM_ENVS"]
            )
            hs_prey = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(teams["prey"]), config["TEST_NUM_ENVS"]
            )
            step_state = (train_states["pred"].params, train_states["prey"].params,
                          env_state, init_obs, init_dones, hs_pred, hs_prey, _rng)
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

    alg_name = config.get("ALG_NAME", "iql_teams")
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
        states = outs["runner_state"][0]  # dict of TrainStates per team
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
        # save metrics as npz for offline plotting
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
