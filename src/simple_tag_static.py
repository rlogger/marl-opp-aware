"""SimpleTagMPE variant with static obstacles and parameterized CW/CCW prey reward shaping.

Two extensions over `simple_tag_v3`:

1. **Static obstacles.** Landmark positions are fixed at construction time and
   reapplied on every reset (including auto-reset on episode termination).
2. **CW/CCW prey shaping.** Per step, an additive reward is given to the prey
   based on the signed angular displacement of the prey relative to each
   obstacle. The shaping is non-zero only when the prey is inside the radial
   annulus [r_in, r_out] around an obstacle.

The shaping bonus per obstacle is:
    bonus = coef * dir_sign * sign(v_prev x v_new) * 1{r_in <= ||v_new|| <= r_out}
where v_prev = prey_pos_{t} - obstacle_pos and v_new = prey_pos_{t+1} - obstacle_pos.
The cross product's z-component is positive for CCW motion and negative for CW.
"""
from functools import partial
from typing import Tuple, Dict, List, Optional

import jax
import jax.numpy as jnp
import chex

from jaxmarl.environments.mpe.simple_tag import SimpleTagMPE
from jaxmarl.environments.mpe.simple import State


class SimpleTagStaticMPE(SimpleTagMPE):
    """SimpleTagMPE with fixed obstacle positions and CW/CCW prey shaping.

    Args:
        obstacle_positions: list of (x, y) coords; length must equal num_obs.
            If None, defaults to [(0.5, 0.5), (-0.5, -0.5)].
        shape_coef: bonus magnitude per obstacle in the active band. 0 disables.
        shape_direction: "ccw" or "cw" — which direction earns the bonus.
        shape_r_in, shape_r_out: radial annulus around each obstacle in which
            the bonus is active.
        **kwargs: forwarded to SimpleTagMPE.
    """

    def __init__(
        self,
        obstacle_positions: Optional[List[List[float]]] = None,
        shape_coef: float = 0.0,
        shape_direction: str = "ccw",
        shape_r_in: float = 0.15,
        shape_r_out: float = 0.6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if obstacle_positions is None:
            obstacle_positions = [[0.5, 0.5], [-0.5, -0.5]]
        positions = jnp.asarray(obstacle_positions, dtype=jnp.float32)
        assert positions.shape == (self.num_landmarks, 2), (
            f"obstacle_positions must be shape ({self.num_landmarks}, 2); got {positions.shape}"
        )
        self.fixed_obstacle_positions = positions

        assert shape_direction in ("ccw", "cw"), shape_direction
        self.shape_coef = float(shape_coef)
        self.shape_dir_sign = 1.0 if shape_direction == "ccw" else -1.0
        self.shape_r_in = float(shape_r_in)
        self.shape_r_out = float(shape_r_out)

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Random agent positions; fixed obstacle positions."""
        key_a, _ = jax.random.split(key)
        agent_pos = jax.random.uniform(
            key_a, (self.num_agents, 2), minval=-1.0, maxval=+1.0
        )
        p_pos = jnp.concatenate([agent_pos, self.fixed_obstacle_positions])
        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents,), False),
            step=0,
        )
        return self.get_obs(state), state

    def _circle_bonus(self, prev_state: State, new_state: State) -> jnp.ndarray:
        """Signed-cross-product CW/CCW bonus for the prey, summed over obstacles."""
        prey_idx = self.num_adversaries  # first good agent
        prev_prey = prev_state.p_pos[prey_idx]                       # (2,)
        new_prey = new_state.p_pos[prey_idx]                         # (2,)
        obs_pos = new_state.p_pos[self.num_agents:]                  # (L, 2)

        v_prev = prev_prey[None, :] - obs_pos                        # (L, 2)
        v_new = new_prey[None, :] - obs_pos                          # (L, 2)
        cross = v_prev[:, 0] * v_new[:, 1] - v_prev[:, 1] * v_new[:, 0]  # (L,)
        dist = jnp.linalg.norm(v_new, axis=-1)                       # (L,)

        in_band = ((dist >= self.shape_r_in) & (dist <= self.shape_r_out)).astype(jnp.float32)
        match = jnp.sign(cross) * self.shape_dir_sign                # +1 / 0 / -1
        return jnp.sum(self.shape_coef * match * in_band)

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        """Delegate to SimpleMPE.step_env, then add the CW/CCW prey shaping.

        Previously this method copy-pasted the parent's body verbatim, which meant any
        upstream change (bug-fix, new world dynamics, extra metadata in info) would
        silently drift. Delegating via ``super().step_env`` keeps us in lock-step with
        JaxMARL: the only thing we own is the single-team reward injection.
        """
        obs, new_state, reward, dones, info = super().step_env(key, state, actions)

        if self.shape_coef != 0.0:
            bonus = self._circle_bonus(state, new_state)
            prey_name = self.good_agents[0]
            reward = {**reward, prey_name: reward[prey_name] + bonus}

        return obs, new_state, reward, dones, info
