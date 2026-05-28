"""Smoke tests for SimpleTagResourcesMPE."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np

from simple_tag_resources import SimpleTagResourcesMPE, ResourceState


OBS = [[0.5, 0.5], [-0.5, -0.5]]
NUM_R = 4


def _make(placement="circle", **kw):
    defaults = dict(
        num_resources=NUM_R,
        placement=placement,
        collect_radius=0.15,
        collect_reward=5.0,
        obstacle_positions=OBS,
    )
    defaults.update(kw)
    return SimpleTagResourcesMPE(**defaults)


def test_circle_placement():
    env = _make("circle")
    _, s = env.reset(jax.random.PRNGKey(0))
    rp = np.asarray(s.resource_pos)
    assert rp.shape == (NUM_R, 2), rp.shape
    dists = np.linalg.norm(rp, axis=-1)
    assert np.allclose(dists, 0.6, atol=1e-5), f"circle radii: {dists}"
    print("PASS circle_placement")


def test_corner_placement():
    env = _make("corners")
    _, s = env.reset(jax.random.PRNGKey(0))
    rp = np.asarray(s.resource_pos)
    expected = np.array([[-0.8, -0.8], [-0.8, 0.8], [0.8, -0.8], [0.8, 0.8]])
    assert np.allclose(rp, expected, atol=1e-5), f"corners: {rp}"
    print("PASS corner_placement")


def test_random_placement_varies():
    env = _make("random")
    placements = set()
    for seed in range(20):
        _, s = env.reset(jax.random.PRNGKey(seed))
        rp = tuple(np.round(np.asarray(s.resource_pos).flatten(), 4))
        placements.add(rp)
    assert len(placements) >= 2, "random placement should produce both types"
    print(f"PASS random_placement_varies ({len(placements)} distinct placements over 20 seeds)")


def test_prey_obs_size():
    env = _make("circle")
    obs, _ = env.reset(jax.random.PRNGKey(0))
    pred_obs = obs[env.adversaries[0]]
    prey_obs = obs[env.good_agents[0]]
    assert pred_obs.shape == (16,), f"pred obs {pred_obs.shape}"
    assert prey_obs.shape == (14 + NUM_R * 3,), f"prey obs {prey_obs.shape}"
    print(f"PASS prey_obs_size (pred={pred_obs.shape}, prey={prey_obs.shape})")


def test_predator_obs_matches_base():
    """Predator observation should be identical to vanilla SimpleTagMPE."""
    env = _make("circle")
    obs, s = env.reset(jax.random.PRNGKey(42))
    pred_obs = np.asarray(obs[env.adversaries[0]])

    from jaxmarl.environments.mpe.simple_tag import SimpleTagMPE
    from jaxmarl.environments.mpe.simple import State as BaseState
    base_env = SimpleTagMPE()
    base_state = BaseState(
        p_pos=s.p_pos, p_vel=s.p_vel, c=s.c, done=s.done, step=s.step,
    )
    base_obs = base_env.get_obs(base_state)
    base_pred = np.asarray(base_obs[base_env.adversaries[0]])
    assert np.allclose(pred_obs, base_pred, atol=1e-5), (
        f"pred obs diverged: max diff = {np.max(np.abs(pred_obs - base_pred))}"
    )
    print("PASS predator_obs_matches_base")


def test_collection_happens():
    """Place prey right on top of a resource, step, verify collection."""
    env = _make("circle", collect_radius=0.2)
    _, s = env.reset(jax.random.PRNGKey(0))
    rp = s.resource_pos[0]
    prey_idx = env.num_adversaries
    p_pos = s.p_pos.at[prey_idx].set(rp)
    s = s.replace(p_pos=p_pos)
    noop = {a: jnp.int32(0) for a in env.agents}
    _, s2, r, _, info = env.step_env(jax.random.PRNGKey(1), s, noop)
    coll = np.asarray(s2.collected)
    assert coll[0], f"resource 0 should be collected, got {coll}"
    assert float(info["resources_collected"]) >= 1.0
    print(f"PASS collection_happens (collected={coll}, info_rc={float(info['resources_collected'])})")


def test_collection_reward():
    """Verify prey reward includes collection bonus when a resource is collected."""
    env = _make("circle", collect_radius=0.2, collect_reward=5.0)
    _, s = env.reset(jax.random.PRNGKey(0))
    noop = {a: jnp.int32(0) for a in env.agents}

    pred_far = jnp.array([-1.3, -1.3])
    base = s.p_pos
    for i in range(env.num_adversaries):
        base = base.at[i].set(pred_far + jnp.array([0.0, 0.05 * i]))

    far_pos = base.at[env.num_adversaries].set(jnp.array([0.0, 0.0]))
    rp0 = s.resource_pos[0]
    near_pos = base.at[env.num_adversaries].set(rp0)

    _, _, r_far, _, _ = env.step_env(jax.random.PRNGKey(1), s.replace(p_pos=far_pos), noop)
    _, _, r_near, _, _ = env.step_env(jax.random.PRNGKey(1), s.replace(p_pos=near_pos), noop)

    prey = env.good_agents[0]
    diff = float(r_near[prey]) - float(r_far[prey])
    assert diff >= 4.5, f"reward diff should be ~5.0, got {diff:.2f}"
    print(f"PASS collection_reward (diff={diff:.2f})")


def test_auto_reset_clears_collected():
    env = _make("circle")
    key = jax.random.PRNGKey(3)
    _, s = env.reset(key)
    s = s.replace(collected=jnp.ones(NUM_R, dtype=bool))
    noop = {a: jnp.int32(0) for a in env.agents}
    for t in range(30):
        key, k = jax.random.split(key)
        _, s, _, _, _ = env.step(k, s, noop)
    coll = np.asarray(s.collected)
    assert not coll.all(), f"after auto-reset, not all should be collected: {coll}"
    print("PASS auto_reset_clears_collected")


def test_fixed_obstacles_preserved():
    env = _make("circle")
    for seed in range(5):
        _, s = env.reset(jax.random.PRNGKey(seed))
        land = np.asarray(s.p_pos[env.num_agents:])
        assert np.allclose(land, np.array(OBS)), f"seed {seed}: {land}"
    print("PASS fixed_obstacles_preserved")


if __name__ == "__main__":
    test_circle_placement()
    test_corner_placement()
    test_random_placement_varies()
    test_prey_obs_size()
    test_predator_obs_matches_base()
    test_collection_happens()
    test_collection_reward()
    test_auto_reset_clears_collected()
    test_fixed_obstacles_preserved()
    print("\nAll smoke tests passed.")
