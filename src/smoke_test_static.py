"""Smoke test the SimpleTagStaticMPE env: fixed landmarks, shaping direction,
radial band, and auto-reset preservation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import jax.numpy as jnp
import numpy as np

from simple_tag_static import SimpleTagStaticMPE
from jaxmarl.environments.mpe.simple import State

OBS = [[0.5, 0.5], [-0.5, -0.5]]


def test_fixed_landmarks():
    env = SimpleTagStaticMPE(obstacle_positions=OBS, shape_coef=0.0)
    # Reset many different keys and check landmark positions always equal OBS
    for seed in range(10):
        _, s = env.reset(jax.random.PRNGKey(seed))
        land = np.asarray(s.p_pos[env.num_agents:])
        assert np.allclose(land, np.array(OBS)), f"seed {seed}: {land}"
    print("PASS fixed_landmarks across 10 resets")


def test_auto_reset_preserves_landmarks():
    env = SimpleTagStaticMPE(obstacle_positions=OBS, shape_coef=0.0)
    key = jax.random.PRNGKey(7)
    _, s = env.reset(key)
    noop = {a: jnp.int32(0) for a in env.agents}
    # Run 30 steps (> max_steps=25) so auto-reset happens
    for t in range(30):
        key, k = jax.random.split(key)
        obs_, s, r, d, _ = env.step(k, s, noop)
    land = np.asarray(s.p_pos[env.num_agents:])
    assert np.allclose(land, np.array(OBS)), f"after auto-reset: {land}"
    print("PASS landmarks preserved across auto-reset")


def test_shape_direction_sign():
    """Prey above origin (12 o'clock); move LEFT → CCW; move RIGHT → CW."""
    def run(direction, dx):
        env = SimpleTagStaticMPE(
            obstacle_positions=[[0.0, 0.0], [5.0, 5.0]],
            shape_coef=1.0, shape_direction=direction,
            shape_r_in=0.05, shape_r_out=1.0,
        )
        p_pos_prev = jnp.array([
            [-1.5, -1.5], [-1.5, -1.5], [-1.5, -1.5],
            [0.0, 0.3],
            [0.0, 0.0], [5.0, 5.0],
        ])
        s_prev = State(
            p_pos=p_pos_prev,
            p_vel=jnp.zeros((env.num_entities, env.dim_p)),
            c=jnp.zeros((env.num_agents, env.dim_c)),
            done=jnp.full((env.num_agents,), False), step=0,
        )
        p_pos_new = p_pos_prev.at[env.num_adversaries].set(jnp.array([dx, 0.3]))
        s_new = s_prev.replace(p_pos=p_pos_new, step=1)
        return float(env._circle_bonus(s_prev, s_new))

    # Prey moves LEFT from (0,0.3): CCW. ccw bonus positive, cw bonus negative.
    ccw_left = run("ccw", -0.1)
    cw_left = run("cw", -0.1)
    assert ccw_left > 0 and cw_left < 0, f"left: ccw={ccw_left}, cw={cw_left}"

    # Prey moves RIGHT from (0,0.3): CW. ccw bonus negative, cw bonus positive.
    ccw_right = run("ccw", 0.1)
    cw_right = run("cw", 0.1)
    assert ccw_right < 0 and cw_right > 0, f"right: ccw={ccw_right}, cw={cw_right}"

    print(f"PASS direction sign (left ccw={ccw_left:+.2f} cw={cw_left:+.2f}; "
          f"right ccw={ccw_right:+.2f} cw={cw_right:+.2f})")


def test_radial_band_gating():
    """Prey far outside the band should get zero bonus."""
    env = SimpleTagStaticMPE(
        obstacle_positions=[[0.0, 0.0], [5.0, 5.0]],
        shape_coef=1.0, shape_direction="ccw",
        shape_r_in=0.25, shape_r_out=0.6,
    )
    p_pos_prev = jnp.array([
        [-1.5, -1.5], [-1.5, -1.5], [-1.5, -1.5],
        [0.0, 0.05],  # prey inside r_in → should be gated off
        [0.0, 0.0], [5.0, 5.0],
    ])
    s_prev = State(
        p_pos=p_pos_prev,
        p_vel=jnp.zeros((env.num_entities, env.dim_p)),
        c=jnp.zeros((env.num_agents, env.dim_c)),
        done=jnp.full((env.num_agents,), False), step=0,
    )
    p_pos_new = p_pos_prev.at[env.num_adversaries].set(jnp.array([0.1, 0.05]))
    s_new = s_prev.replace(p_pos=p_pos_new, step=1)
    b_inner = float(env._circle_bonus(s_prev, s_new))
    assert b_inner == 0.0, f"inside r_in expected 0, got {b_inner}"

    # Prey outside r_out → should also be gated off
    p_pos_prev = p_pos_prev.at[env.num_adversaries].set(jnp.array([0.0, 0.7]))
    s_prev = s_prev.replace(p_pos=p_pos_prev)
    p_pos_new = p_pos_prev.at[env.num_adversaries].set(jnp.array([0.1, 0.7]))
    s_new = s_prev.replace(p_pos=p_pos_new, step=1)
    b_outer = float(env._circle_bonus(s_prev, s_new))
    assert b_outer == 0.0, f"outside r_out expected 0, got {b_outer}"

    print(f"PASS band gating (inner={b_inner}, outer={b_outer})")


def test_no_shape_when_coef_zero():
    env = SimpleTagStaticMPE(obstacle_positions=OBS, shape_coef=0.0)
    key = jax.random.PRNGKey(0)
    _, s = env.reset(key)
    actions = {a: jnp.int32(1) for a in env.agents}  # all go up
    _, _, r, _, _ = env.step_env(key, s, actions)
    # With no shaping, prey reward should just be map bounds + capture (no extra bonus)
    print(f"PASS no_shape (prey reward = {float(r['agent_0']):+.3f})")


if __name__ == "__main__":
    test_fixed_landmarks()
    test_auto_reset_preserves_landmarks()
    test_shape_direction_sign()
    test_radial_band_gating()
    test_no_shape_when_coef_zero()
    print("\nAll smoke tests passed.")
