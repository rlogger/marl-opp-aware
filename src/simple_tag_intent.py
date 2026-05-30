"""SimpleTagMPE with a HIDDEN prey intent (the opponent-modeling task).

Unlike the resource variant -- where the prey's "strategy" is induced by changing
the environment (circle vs corners layout) and is therefore readable from the
observation -- here the strategy is an intrinsic, hidden property of the opponent:

  * At every reset the prey is assigned an intent  g in {0,..,K-1}, one of K
    fixed corners. The prey earns a small per-step bonus for loitering within
    `intent_radius` of corner g (in addition to evading), so it learns to haunt
    its assigned corner when it is safe to.
  * The prey observes its own intent (one-hot appended to its obs). The predators
    do NOT (unless `reveal_to_pred=True`, the oracle condition). The environment
    -- obstacles, corners, dynamics -- is identical across intents.

So a predator cannot read the prey's strategy off a single observation; it must
infer it from the prey's motion over time. This is the setting in which opponent
modeling earns its keep:

  * unaware predator  (reveal_to_pred=False): must hedge over the 4 corners.
  * oracle  predator  (reveal_to_pred=True) : is told g, can pre-position.
  * inferred predator : oracle architecture fed an intent inferred from the
    prey's trajectory (see part2_intent_eval.py).

Predators are optionally slowed (`pred_max_speed/accel`) so the prey is not
dominated and can actually express its intent.
"""
from functools import partial
from typing import List, Optional

import jax
import jax.numpy as jnp
import chex
from flax import struct

from jaxmarl.environments.mpe.simple_tag import SimpleTagMPE
from jaxmarl.environments.spaces import Box


@struct.dataclass
class IntentState:
    p_pos: chex.Array
    p_vel: chex.Array
    c: chex.Array
    done: chex.Array
    step: int
    goal: int = None
    intent: int = None
    belief_alpha: float = None      # sharpness of the predator's intent belief


class SimpleTagIntentMPE(SimpleTagMPE):
    def __init__(
        self,
        num_intents: int = 4,
        corner_offset: float = 0.8,
        intent_radius: float = 0.45,
        intent_reward: float = 1.0,
        reveal_to_pred: bool = False,
        intent_belief_noise: bool = False,
        obstacle_positions: Optional[List[List[float]]] = None,
        pred_max_speed: Optional[float] = None,
        pred_accel: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if pred_max_speed is not None:
            self.max_speed = self.max_speed.at[:self.num_adversaries].set(float(pred_max_speed))
        if pred_accel is not None:
            self.accel = self.accel.at[:self.num_adversaries].set(float(pred_accel))

        self.num_intents = int(num_intents)
        self.intent_radius = float(intent_radius)
        self.intent_reward = float(intent_reward)
        self.reveal_to_pred = bool(reveal_to_pred)
        self.intent_belief_noise = bool(intent_belief_noise)

        # K fixed corners (K=4 default). For K<4, take the first K.
        c = float(corner_offset)
        all_corners = jnp.array([[-c, -c], [-c, c], [c, -c], [c, c]])
        self.corners = all_corners[: self.num_intents]

        if obstacle_positions is not None:
            pos = jnp.asarray(obstacle_positions, dtype=jnp.float32)
            assert pos.shape == (self.num_landmarks, 2)
            self.fixed_obstacle_positions = pos
        else:
            self.fixed_obstacle_positions = None

        # obs dims: base + one-hot intent (prey always; predators iff revealed)
        base_prey = self.observation_spaces[self.good_agents[0]].shape[0]
        for a in self.good_agents:
            self.observation_spaces[a] = Box(-jnp.inf, jnp.inf,
                                             (base_prey + self.num_intents,))
        if self.reveal_to_pred:
            base_pred = self.observation_spaces[self.adversaries[0]].shape[0]
            for a in self.adversaries:
                self.observation_spaces[a] = Box(-jnp.inf, jnp.inf,
                                                 (base_pred + self.num_intents,))

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey):
        key_a, key_l, key_i, key_b = jax.random.split(key, 4)
        agent_pos = jax.random.uniform(key_a, (self.num_agents, 2), minval=-1.0, maxval=1.0)
        if self.fixed_obstacle_positions is not None:
            landmark_pos = self.fixed_obstacle_positions
        else:
            landmark_pos = jax.random.uniform(key_l, (self.num_landmarks, 2), minval=-1.0, maxval=1.0)
        p_pos = jnp.concatenate([agent_pos, landmark_pos])
        intent = jax.random.randint(key_i, (), 0, self.num_intents)
        # per-episode belief sharpness: 1 = perfect intent, 0 = flat (no info).
        belief_alpha = jnp.where(self.intent_belief_noise,
                                 jax.random.uniform(key_b), 1.0)

        state = IntentState(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents,), False),
            step=0,
            intent=intent,
            belief_alpha=belief_alpha,
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: IntentState, actions: dict):
        obs, new_state, reward, dones, info = super().step_env(key, state, actions)

        prey_idx = self.num_adversaries
        prey_pos = new_state.p_pos[prey_idx]
        target = self.corners[new_state.intent]
        dist = jnp.linalg.norm(prey_pos - target)
        at_corner = dist < self.intent_radius
        bonus = self.intent_reward * at_corner.astype(jnp.float32)

        prey_name = self.good_agents[0]
        reward = {**reward, prey_name: reward[prey_name] + bonus}

        obs = self.get_obs(new_state)
        info["intent"] = new_state.intent
        info["at_corner"] = at_corner.astype(jnp.float32)
        return obs, new_state, reward, dones, info

    def get_obs(self, state) -> dict:
        base = super().get_obs(state)
        onehot = jax.nn.one_hot(state.intent, self.num_intents)
        for a in self.good_agents:                      # prey always knows its intent
            base[a] = jnp.concatenate([base[a], onehot])
        if self.reveal_to_pred:
            if self.intent_belief_noise:
                uniform = jnp.full((self.num_intents,), 1.0 / self.num_intents)
                belief = state.belief_alpha * onehot + (1.0 - state.belief_alpha) * uniform
            else:
                belief = onehot
            for a in self.adversaries:
                base[a] = jnp.concatenate([base[a], belief])
        return base
