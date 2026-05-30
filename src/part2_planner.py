"""Part 2: a belief-conditioned Monte-Carlo planner for the predators.

The belief predator of Part 1 (part2_intent_eval.py) is reactive. Here the
predators *plan*: at each step they hold the online intent posterior b_t and, for
each candidate joint predator action, roll the true simulator forward H steps
with the prey's moves sampled from its own policy under an intent g ~ b_t. They
pick the joint action that maximizes expected discounted captures. This is the
proposal's Part 2 -- sampling the opponent's moves from a belief-conditioned
model inside a lookahead -- using the real simulator as the dynamics model.

Variance control: the sampled intents g are drawn once per (episode, rollout) and
shared across all candidate actions, so candidates are compared on identical prey
realizations (a paired comparison), which makes the arg-max over candidates
low-variance even with few rollouts.

We compare captures/episode for: planner (ours), reactive belief, unaware, oracle.

Usage:
    python src/part2_planner.py
"""
import itertools
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_trajectory_dataset_resources import ActorLogits
from jaxmarl.wrappers.baselines import load_params
import part2_intent_eval as PE

PLOTDIR = "plots"
LOGDIR = "logs/MPE_simple_tag_v3"
TABLE = jnp.array(list(itertools.product(range(5), repeat=3)))   # (125,3) joint actions
C = TABLE.shape[0]
B = 32                  # parallel episodes
KROLL = 8               # rollouts per candidate (shared intents)
H = 5                   # planning horizon
GAMMA = 0.95
W_LEAF = 1.0            # leaf shaping: reward ending close to the imagined prey
NET = ActorLogits(action_dim=5, hidden_dim=128)


def _slots(env):
    sizes = {a: env.observation_space(a).shape[0] for a in env.agents}
    max_obs = max(sizes.values())
    prey_base = sizes[env.good_agents[0]]
    pred_base = sizes[env.adversaries[0]]
    return max_obs, (prey_base - PE.K, prey_base), (pred_base - PE.K, pred_base)


def planner_action(env, state, b_t, prey_p, belief_p, rng, slots):
    """Return (B,3) predator actions chosen by belief-conditioned MC lookahead."""
    max_obs, pisl, isl = slots
    advs = env.adversaries; prey = env.good_agents[0]
    N = B * C * KROLL

    tile = lambda x: jnp.broadcast_to(
        x[:, None, None], (B, C, KROLL) + x.shape[1:]).reshape((N,) + x.shape[1:])
    st = jax.tree_util.tree_map(tile, state)

    # shared intents g ~ b_t, drawn per (episode, rollout), broadcast over candidates
    rng, kg = jax.random.split(rng)
    g = jax.random.categorical(kg, jnp.log(b_t + 1e-9)[:, None, :],
                               shape=(B, KROLL))                      # (B,KROLL)
    g_oh = jax.nn.one_hot(g, PE.K)                                    # (B,KROLL,K)
    g_oh = jnp.broadcast_to(g_oh[:, None], (B, C, KROLL, PE.K)).reshape(N, PE.K)
    bel = jnp.broadcast_to(b_t[:, None, None, :], (B, C, KROLL, PE.K)).reshape(N, PE.K)
    root = jnp.broadcast_to(TABLE[None, :, None, :], (B, C, KROLL, 3)).reshape(N, 3)

    get_obs = jax.vmap(env.get_obs)
    step = jax.vmap(env.step_env)
    obs = get_obs(st)
    ret = jnp.zeros(N); disc = 1.0
    for h in range(H):
        prey_in = PE._pad(obs[prey], max_obs).at[:, pisl[0]:pisl[1]].set(g_oh)
        a_prey = NET.apply(prey_p, prey_in).argmax(-1).astype(jnp.int32)
        if h == 0:
            a_pred = root
        else:
            cols = []
            for a in advs:
                pin = PE._pad(obs[a], max_obs).at[:, isl[0]:isl[1]].set(bel)
                cols.append(NET.apply(belief_p, pin).argmax(-1).astype(jnp.int32))
            a_pred = jnp.stack(cols, -1)
        acts = {prey: a_prey}
        for j, a in enumerate(advs):
            acts[a] = a_pred[:, j]
        rng, ks = jax.random.split(rng)
        obs, st, rew, _, _ = step(jax.random.split(ks, N), st, acts)
        ret = ret + disc * (rew[advs[0]] / 10.0)
        disc *= GAMMA
    # dense leaf term: prefer ending close to the imagined future prey position
    # (under a sampled intent the prey heads to the believed corner), i.e. plan
    # an interception rather than relying on the rare in-horizon capture.
    pp = st.p_pos[:, :env.num_adversaries]            # (N,3,2)
    qp = st.p_pos[:, env.num_adversaries]             # (N,2)
    leaf = jnp.linalg.norm(pp - qp[:, None], axis=-1).min(-1)   # nearest predator
    ret = ret - W_LEAF * leaf
    score = ret.reshape(B, C, KROLL).mean(-1)        # (B,C)
    best = score.argmax(-1)                           # (B,)
    return TABLE[best]                                # (B,3)


def run_planner_episode(env, prey_p, belief_p, enc_ks, enc_proba, rng,
                        belief_mode="online"):
    """Full real episode where predators act by the planner; prey is the fixed
    belief-condition prey. `belief_mode`: 'online' feeds the encoder's
    sharpening posterior; 'flat' forces a uniform belief (ablation: lookahead
    without opponent inference). Returns captures (B,)."""
    slots = _slots(env); max_obs = slots[0]
    advs = env.adversaries; prey = env.good_agents[0]
    rng, kr = jax.random.split(rng)
    obs, state = jax.vmap(env.reset)(jax.random.split(kr, B))
    step = jax.vmap(env.step_env)
    prey_xy = [np.asarray(state.p_pos[:, env.num_adversaries])]
    cap = np.zeros(B); cache = {}
    for t in range(env.max_steps):
        if belief_mode == "flat":
            b_t = jnp.full((B, PE.K), 1.0 / PE.K)
        else:
            usable = [k for k in enc_ks if k <= len(prey_xy)]
            if not usable:
                b_t = jnp.full((B, PE.K), 1.0 / PE.K)
            else:
                k = usable[-1]
                if k not in cache:
                    cache[k] = jnp.asarray(enc_proba[k](np.stack(prey_xy[:k], 1)))
                b_t = cache[k]
        rng, kp = jax.random.split(rng)
        a_pred = planner_action(env, state, b_t, prey_p, belief_p, kp, slots)  # (B,3)
        a_prey = NET.apply(prey_p, PE._pad(obs[prey], max_obs)).argmax(-1).astype(jnp.int32)
        acts = {prey: a_prey}
        for j, a in enumerate(advs):
            acts[a] = a_pred[:, j]
        rng, ks = jax.random.split(rng)
        obs, state, rew, _, info = step(jax.random.split(ks, B), state, acts)
        prey_xy.append(np.asarray(state.p_pos[:, env.num_adversaries]))
        cap += np.asarray(rew[advs[0]]) / 10.0
    return cap


def main():
    os.makedirs(PLOTDIR, exist_ok=True)
    env = PE.make_env(reveal=True)

    # encoder (soft) for the online belief
    pos, intent = [], []
    for i in PE.SEEDS:
        P, g = PE.gather_prey_traj(env,
                                   load_params(PE.apath("mappo_intent_oracle", "prey", i)),
                                   load_params(PE.apath("mappo_intent_oracle", "pred", i)),
                                   120, jax.random.PRNGKey(i))
        pos.append(P); intent.append(g)
    pos = np.concatenate(pos); intent = np.concatenate(intent)
    ks = [3, 5, 8, 12, 18, 25]; enc_proba = {}
    for k in ks:
        _, _, _, soft = PE.train_encoder_at_k(pos, intent, k)
        enc_proba[k] = soft

    planner_caps, flat_caps = [], []
    for i in PE.SEEDS:
        prey_be = load_params(PE.apath("mappo_intent_belief", "prey", i))
        pred_be = load_params(PE.apath("mappo_intent_belief", "pred", i))
        cap = run_planner_episode(env, prey_be, pred_be, ks, enc_proba,
                                  jax.random.PRNGKey(200 + i), belief_mode="online")
        capf = run_planner_episode(env, prey_be, pred_be, ks, enc_proba,
                                   jax.random.PRNGKey(200 + i), belief_mode="flat")
        planner_caps.append(cap.mean()); flat_caps.append(capf.mean())
        print(f"  seed {i}: planner(online) {cap.mean():.2f}  "
              f"planner(flat) {capf.mean():.2f}", flush=True)
    pl_mean, pl_std = float(np.mean(planner_caps)), float(np.std(planner_caps))
    fl_mean, fl_std = float(np.mean(flat_caps)), float(np.std(flat_caps))

    # pull the reactive/oracle/unaware numbers from the Part-1 eval npz
    prev = np.load(os.path.join(LOGDIR, "part2_intent_eval.npz"))
    ref = {m: (float(prev[f"cap_{m}"].mean()), float(prev[f"cap_{m}"].std()))
           for m in ["unaware", "belief", "oracle"]}
    print("\ncaptures/episode:")
    print(f"  unaware          : {ref['unaware'][0]:.2f}")
    print(f"  reactive belief  : {ref['belief'][0]:.2f}")
    print(f"  planner (flat b) : {fl_mean:.2f} +/- {fl_std:.2f}   <- ablation")
    print(f"  PLANNER (online) : {pl_mean:.2f} +/- {pl_std:.2f}")
    print(f"  oracle (reactive): {ref['oracle'][0]:.2f}")
    gain = 100 * (pl_mean - ref["belief"][0]) / ref["belief"][0]
    abl = 100 * (pl_mean - fl_mean) / fl_mean
    print(f"planner(online) vs reactive belief: {gain:+.0f}%")
    print(f"planner(online) vs planner(flat):   {abl:+.0f}%  (the belief's contribution)")

    np.savez(os.path.join(LOGDIR, "part2_planner.npz"),
             planner=np.array(planner_caps), planner_flat=np.array(flat_caps),
             **{f"ref_{m}": np.array(prev[f"cap_{m}"])
                for m in ["unaware", "belief", "oracle"]},
             B=B, C=C, KROLL=KROLL, H=H)

    # figure
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    labels = ["unaware", "reactive\nbelief", "planner\n(flat belief)",
              "planner\n(online belief)", "oracle\n(reactive)"]
    means = [ref["unaware"][0], ref["belief"][0], fl_mean, pl_mean, ref["oracle"][0]]
    stds = [ref["unaware"][1], ref["belief"][1], fl_std, pl_std, ref["oracle"][1]]
    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                  color=["#999", "#1f7a4e", "#c0a0a0", "#0d6e7a", "#073b42"])
    ax.set_ylabel("captures / episode")
    ax.set_title(f"Part 2 — planning with a belief-conditioned opponent model\n"
                 f"online-belief planner {pl_mean:.2f} vs reactive {ref['belief'][0]:.2f} "
                 f"(+{gain:.0f}%); the belief adds +{abl:.0f}% over flat")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.05, f"{m:.2f}",
                ha="center", fontweight="bold", fontsize=9)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(PLOTDIR, "part2_planner.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
