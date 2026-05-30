"""Part 3: plan with a LEARNED dynamics model -- no simulator at test time.

Part 2 planned through the true simulator. Here we learn the physics transition
of the four agents,
    (pos, vel, joint actions) -> (pos', vel'),
and run the same belief-conditioned planner through the learned model instead of
env.step_env. The observation and capture readouts stay geometric functions of
the predicted state (reused env.get_obs / a distance test), so the only learned
component is the dynamics -- the hard part. If the learned-model planner recovers
the true-simulator planner's captures, we can plan without the simulator.

We report (i) the dynamics model's k-step position error and (ii) the capture
ladder with the learned-model planner alongside the true-sim planner.

Usage:
    python src/part3_learned_planner.py
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
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
from flax.training.train_state import TrainState
import optax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_trajectory_dataset_resources import ActorLogits
from jaxmarl.wrappers.baselines import load_params
import part2_intent_eval as PE

PLOTDIR = "plots"; LOGDIR = "logs/MPE_simple_tag_v3"
TABLE = jnp.array(list(itertools.product(range(5), repeat=3))); C = TABLE.shape[0]
B, KROLL, H, GAMMA, W_LEAF = 32, 8, 5, 0.95, 1.0
H_PLAN = 2                   # short model lookahead; a learned value carries the rest
NET = ActorLogits(action_dim=5, hidden_dim=128)
NA = 4                       # agents (3 pred + 1 prey)


class Dyn(nn.Module):
    @nn.compact
    def __call__(self, x):                       # x:(.,4*4 + 4*5)=36 -> (.,16) residual
        h = nn.relu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)))(x))
        h = nn.relu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)))(h))
        return nn.Dense(16, kernel_init=orthogonal(0.01))(h)


class Val(nn.Module):
    """Belief-conditioned value: captures-to-go under the belief policy."""
    @nn.compact
    def __call__(self, feat, bel):               # feat:(.,16) agent pos+vel, bel:(.,4)
        x = jnp.concatenate([feat, bel], -1)
        x = nn.relu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)))(x))
        x = nn.relu(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)))(x))
        return nn.Dense(1, kernel_init=orthogonal(0.01))(x)[..., 0]


def state_feat(state):
    ap, av = agent_state(state)
    return jnp.concatenate([ap.reshape(*ap.shape[:-2], 8),
                            av.reshape(*av.shape[:-2], 8)], -1)


def _slots(env):
    sizes = {a: env.observation_space(a).shape[0] for a in env.agents}
    max_obs = max(sizes.values())
    prey_base = sizes[env.good_agents[0]]; pred_base = sizes[env.adversaries[0]]
    return max_obs, (prey_base - PE.K, prey_base), (pred_base - PE.K, pred_base)


def agent_state(state):
    """(pos, vel) of the 4 agents from an env state -> (.,4,2),(.,4,2)."""
    return state.p_pos[..., :NA, :], state.p_vel[..., :NA, :]


def dyn_input(apos, avel, acts_oh):
    return jnp.concatenate([apos.reshape(*apos.shape[:-2], 8),
                            avel.reshape(*avel.shape[:-2], 8),
                            acts_oh.reshape(*acts_oh.shape[:-2], 20)], -1)


# ---------------------------------------------------------------- data + train #
def collect_seq(env, prey_p, pred_p, n_envs, steps, rng, eps=0.5):
    """Roll the env keeping per-env sequences: agent pos/vel and the joint actions
    taken at each step (for multi-step / rollout training)."""
    sizes = {a: env.observation_space(a).shape[0] for a in env.agents}
    max_obs = max(sizes.values())
    advs = env.adversaries; prey = env.good_agents[0]
    reset = jax.vmap(env.reset); step = jax.vmap(env.step_env)
    rng, kr = jax.random.split(rng); obs, st = reset(jax.random.split(kr, n_envs))
    ap, av = agent_state(st)
    pos_log = [np.asarray(ap)]; vel_log = [np.asarray(av)]; act_log = []
    for t in range(steps):
        rng, ka, kp = jax.random.split(rng, 3)
        ag = NET.apply(prey_p, PE._pad(obs[prey], max_obs)).argmax(-1)
        rnd = jax.random.randint(ka, (n_envs,), 0, 5)
        coin = jax.random.bernoulli(kp, eps, (n_envs,))
        acts = {prey: jnp.where(coin, rnd, ag).astype(jnp.int32)}
        for a in advs:
            rng, kk, kc = jax.random.split(rng, 3)
            gg = NET.apply(pred_p, PE._pad(obs[a], max_obs)).argmax(-1)
            rr = jax.random.randint(kk, (n_envs,), 0, 5)
            acts[a] = jnp.where(jax.random.bernoulli(kc, eps, (n_envs,)), rr, gg).astype(jnp.int32)
        act_log.append(np.asarray(jnp.stack([acts[a] for a in (advs + [prey])], 1)))  # (n,4)
        rng, ks = jax.random.split(rng)
        obs, st, _, _, _ = step(jax.random.split(ks, n_envs), st, acts)
        ap, av = agent_state(st)
        pos_log.append(np.asarray(ap)); vel_log.append(np.asarray(av))
    return (np.stack(pos_log, 1), np.stack(vel_log, 1), np.stack(act_log, 1))


def train_dynamics(pos, vel, acts, L=4, steps=12000, seed=0):
    """Rollout training: unroll the model L steps from a window's start using the
    recorded actions and match the true L-step pos/vel, so the model is accurate
    over a horizon (not just one step) and compounding error stays small."""
    n_envs, T1 = pos.shape[0], pos.shape[1]; T = T1 - 1
    res = np.concatenate([(pos[:, 1:] - pos[:, :-1]).reshape(-1, 8),
                          (vel[:, 1:] - vel[:, :-1]).reshape(-1, 8)], -1)
    ym, ys = res.mean(0), res.std(0) + 1e-6
    net = Dyn(); rng = jax.random.PRNGKey(seed)
    params = net.init(rng, jnp.zeros((1, 36)))
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    stt = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
    posj, velj, actj = jnp.asarray(pos), jnp.asarray(vel), jnp.asarray(acts)
    ymj, ysj = jnp.asarray(ym), jnp.asarray(ys)

    @jax.jit
    def upd(stt, ei, t0):
        def loss(pp):
            ap0 = posj[ei, t0]; av0 = velj[ei, t0]
            aw = actj[ei[:, None], t0[:, None] + jnp.arange(L)[None, :]]    # (b,L,4)
            aw_oh = jax.nn.one_hot(aw, 5)                                   # (b,L,4,5)

            def fstep(carry, l):
                ap, av = carry
                d = net.apply(pp, dyn_input(ap, av, aw_oh[:, l])) * ysj + ymj
                ap = ap + d[..., :8].reshape(ap.shape)
                av = av + d[..., 8:].reshape(av.shape)
                return (ap, av), (ap, av)
            _, (aps, avs) = jax.lax.scan(fstep, (ap0, av0), jnp.arange(L))
            aps = aps.transpose(1, 0, 2, 3); avs = avs.transpose(1, 0, 2, 3)
            tp = posj[ei[:, None], t0[:, None] + 1 + jnp.arange(L)[None, :]]
            tv = velj[ei[:, None], t0[:, None] + 1 + jnp.arange(L)[None, :]]
            return jnp.mean((aps - tp) ** 2) + 0.3 * jnp.mean((avs - tv) ** 2)
        l, g = jax.value_and_grad(loss)(stt.params)
        return stt.apply_gradients(grads=g), l

    for s in range(steps):
        rng, k1, k2 = jax.random.split(rng, 3)
        ei = jax.random.randint(k1, (256,), 0, n_envs)
        t0 = jax.random.randint(k2, (256,), 0, T - L)
        stt, l = upd(stt, ei, t0)
        if s % 3000 == 0 or s == steps - 1:
            print(f"    dyn step {s:5d}  {L}-step mse {float(l):.5f}", flush=True)

    def predict(apos, avel, acts_oh):
        d = net.apply(stt.params, dyn_input(apos, avel, acts_oh)) * ysj + ymj
        return apos + d[..., :8].reshape(*apos.shape), avel + d[..., 8:].reshape(*avel.shape)
    return predict


def collect_value(env, prey_p, belief_p, ks_enc, enc_proba, n_envs, rng, n_batches=4):
    """Roll the reactive belief policy and record (agent-state, belief,
    captures-to-go) for fitting the belief-conditioned value V(s,b)."""
    slots = _slots(env); max_obs, _, isl = slots
    advs = env.adversaries; prey = env.good_agents[0]
    step = jax.vmap(env.step_env)
    Xf, Xb, R = [], [], []
    for _ in range(n_batches):
        rng, kr = jax.random.split(rng)
        obs, st = jax.vmap(env.reset)(jax.random.split(kr, n_envs))
        prey_xy = [np.asarray(st.p_pos[:, env.num_adversaries])]
        feats, bels, caps = [], [], []; cache = {}
        for t in range(env.max_steps):
            usable = [k for k in ks_enc if k <= len(prey_xy)]
            if not usable:
                b_t = jnp.full((n_envs, PE.K), 1.0 / PE.K)
            else:
                k = usable[-1]
                if k not in cache:
                    cache[k] = jnp.asarray(enc_proba[k](np.stack(prey_xy[:k], 1)))
                b_t = cache[k]
            feats.append(np.asarray(state_feat(st))); bels.append(np.asarray(b_t))
            a_prey = NET.apply(prey_p, PE._pad(obs[prey], max_obs)).argmax(-1).astype(jnp.int32)
            acts = {prey: a_prey}
            for a in advs:
                pin = PE._pad(obs[a], max_obs).at[:, isl[0]:isl[1]].set(b_t)
                acts[a] = NET.apply(belief_p, pin).argmax(-1).astype(jnp.int32)
            rng, ks = jax.random.split(rng)
            obs, st, rew, _, _ = step(jax.random.split(ks, n_envs), st, acts)
            caps.append(np.asarray(rew[advs[0]]) / 10.0)
            prey_xy.append(np.asarray(st.p_pos[:, env.num_adversaries]))
        caps = np.stack(caps, 1)                          # (n,T)
        rtg = np.zeros_like(caps); acc = np.zeros(n_envs)
        for t in range(caps.shape[1] - 1, -1, -1):
            acc = caps[:, t] + GAMMA * acc; rtg[:, t] = acc
        Xf.append(np.stack(feats, 1).reshape(-1, 16))
        Xb.append(np.stack(bels, 1).reshape(-1, PE.K))
        R.append(rtg.reshape(-1))
    return np.concatenate(Xf), np.concatenate(Xb), np.concatenate(R)


def train_value(Xf, Xb, R, steps=6000, seed=0):
    net = Val(); rng = jax.random.PRNGKey(seed)
    params = net.init(rng, jnp.zeros((1, 16)), jnp.zeros((1, PE.K)))
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    stt = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
    Xfj, Xbj, Rj = jnp.asarray(Xf), jnp.asarray(Xb), jnp.asarray(R); n = len(R)

    @jax.jit
    def upd(stt, idx):
        def loss(pp):
            return jnp.mean((net.apply(pp, Xfj[idx], Xbj[idx]) - Rj[idx]) ** 2)
        l, g = jax.value_and_grad(loss)(stt.params)
        return stt.apply_gradients(grads=g), l
    for s in range(steps):
        rng, bk = jax.random.split(rng)
        idx = jax.random.randint(bk, (512,), 0, n)
        stt, l = upd(stt, idx)
        if s % 2000 == 0 or s == steps - 1:
            print(f"    val step {s:5d}  mse {float(l):.4f}", flush=True)
    return lambda feat, bel: net.apply(stt.params, feat, bel)


# ----------------------------------------------------------- learned planner #
def rebuild_state(template, apos, avel):
    """Put predicted agent pos/vel back into a (batched) IntentState, keeping the
    fixed landmarks and the carried intent."""
    n = apos.shape[0]
    land_pos = template.p_pos[:, NA:]; land_vel = template.p_vel[:, NA:]
    p_pos = jnp.concatenate([apos, land_pos], 1)
    p_vel = jnp.concatenate([avel, land_vel], 1)
    return template.replace(p_pos=p_pos, p_vel=p_vel)


def captures_from(state, env):
    pp = state.p_pos[:, :env.num_adversaries]; qp = state.p_pos[:, env.num_adversaries]
    d = jnp.linalg.norm(pp - qp[:, None], axis=-1)
    rad = float(env.rad[0]) + float(env.rad[env.num_adversaries])
    return (d < rad).sum(-1).astype(jnp.float32)        # # predators in contact


def planner_action(env, state, b_t, prey_p, belief_p, predict, value_fn, rng, slots):
    """H-step lookahead through the learned dynamics with a distance leaf (the
    learned-value leaf was tried and hurt -- see notes). `value_fn` is accepted
    for interface parity but unused in the reported configuration."""
    max_obs, pisl, isl = slots
    advs = env.adversaries; prey = env.good_agents[0]; N = B * C * KROLL
    tile = lambda x: jnp.broadcast_to(x[:, None, None],
                                      (B, C, KROLL) + x.shape[1:]).reshape((N,) + x.shape[1:])
    st = jax.tree_util.tree_map(tile, state)
    rng, kg = jax.random.split(rng)
    g = jax.random.categorical(kg, jnp.log(b_t + 1e-9)[:, None, :], shape=(B, KROLL))
    g_oh = jnp.broadcast_to(jax.nn.one_hot(g, PE.K)[:, None],
                            (B, C, KROLL, PE.K)).reshape(N, PE.K)
    bel = jnp.broadcast_to(b_t[:, None, None, :], (B, C, KROLL, PE.K)).reshape(N, PE.K)
    root = jnp.broadcast_to(TABLE[None, :, None, :], (B, C, KROLL, 3)).reshape(N, 3)
    get_obs = jax.vmap(env.get_obs)
    ret = jnp.zeros(N); disc = 1.0
    for h in range(H):
        obs = get_obs(st)
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
        acts_oh = jnp.stack([jax.nn.one_hot(a_pred[:, 0], 5), jax.nn.one_hot(a_pred[:, 1], 5),
                             jax.nn.one_hot(a_pred[:, 2], 5), jax.nn.one_hot(a_prey, 5)], 1)
        apos, avel = agent_state(st)
        napos, navel = predict(apos, avel, acts_oh)          # LEARNED dynamics
        st = rebuild_state(st, napos, navel)
        ret = ret + disc * (captures_from(st, env))
        disc *= GAMMA
    pp = st.p_pos[:, :env.num_adversaries]; qp = st.p_pos[:, env.num_adversaries]
    ret = ret - W_LEAF * jnp.linalg.norm(pp - qp[:, None], axis=-1).min(-1)
    best = ret.reshape(B, C, KROLL).mean(-1).argmax(-1)
    return TABLE[best]


def run_episode(env, prey_p, belief_p, predict, value_fn, ks_enc, enc_proba, rng):
    slots = _slots(env); max_obs = slots[0]
    advs = env.adversaries; prey = env.good_agents[0]
    rng, kr = jax.random.split(rng)
    obs, state = jax.vmap(env.reset)(jax.random.split(kr, B))
    step = jax.vmap(env.step_env)
    prey_xy = [np.asarray(state.p_pos[:, env.num_adversaries])]; cap = np.zeros(B); cache = {}
    for t in range(env.max_steps):
        usable = [k for k in ks_enc if k <= len(prey_xy)]
        if not usable:
            b_t = jnp.full((B, PE.K), 1.0 / PE.K)
        else:
            k = usable[-1]
            if k not in cache:
                cache[k] = jnp.asarray(enc_proba[k](np.stack(prey_xy[:k], 1)))
            b_t = cache[k]
        rng, kp = jax.random.split(rng)
        a_pred = planner_action(env, state, b_t, prey_p, belief_p, predict, value_fn, kp, slots)
        a_prey = NET.apply(prey_p, PE._pad(obs[prey], max_obs)).argmax(-1).astype(jnp.int32)
        acts = {prey: a_prey}
        for j, a in enumerate(advs):
            acts[a] = a_pred[:, j]
        rng, kstep = jax.random.split(rng)
        obs, state, rew, _, _ = step(jax.random.split(kstep, B), state, acts)
        prey_xy.append(np.asarray(state.p_pos[:, env.num_adversaries]))
        cap += np.asarray(rew[advs[0]]) / 10.0
    return cap


def main():
    os.makedirs(PLOTDIR, exist_ok=True)
    env = PE.make_env(reveal=True)

    # 1. learn the dynamics (rollout training on sequential data)
    print("collecting transitions...")
    pos, vel, acts = collect_seq(env, load_params(PE.apath("mappo_intent_belief", "prey", 0)),
                                 load_params(PE.apath("mappo_intent_belief", "pred", 0)),
                                 n_envs=512, steps=200, rng=jax.random.PRNGKey(0))
    print(f"  {pos.shape[0] * (pos.shape[1]-1)} transitions, seq len {pos.shape[1]}")
    predict = train_dynamics(pos, vel, acts, L=4, steps=12000)

    # 2. closed-loop H-step position error on held-out greedy rollouts
    tp, tv, ta = collect_seq(env, load_params(PE.apath("mappo_intent_belief", "prey", 1)),
                             load_params(PE.apath("mappo_intent_belief", "pred", 1)),
                             n_envs=128, steps=26, rng=jax.random.PRNGKey(99), eps=0.0)
    ap, av = jnp.asarray(tp[:, 0]), jnp.asarray(tv[:, 0])
    errs = []
    for h in range(min(H, ta.shape[1])):
        ap, av = predict(ap, av, jax.nn.one_hot(jnp.asarray(ta[:, h]), 5))
        errs.append(float(np.sqrt(((np.asarray(ap) - tp[:, h + 1]) ** 2).mean())))
    err1, errH = errs[0], errs[-1]
    print(f"  dynamics pos RMSE: 1-step {err1:.4f}, {len(errs)}-step {errH:.4f} "
          f"(arena 3.2 wide)")

    # 3. encoder for the online belief
    pos, intent = [], []
    for i in PE.SEEDS:
        P, g = PE.gather_prey_traj(env, load_params(PE.apath("mappo_intent_oracle", "prey", i)),
                                   load_params(PE.apath("mappo_intent_oracle", "pred", i)),
                                   120, jax.random.PRNGKey(i))
        pos.append(P); intent.append(g)
    pos = np.concatenate(pos); intent = np.concatenate(intent)
    ks = [3, 5, 8, 12, 18, 25]; enc_proba = {}
    for k in ks:
        _, _, _, soft = PE.train_encoder_at_k(pos, intent, k)
        enc_proba[k] = soft

    # 4. learned-model planner (learned dynamics, H-step lookahead, distance leaf).
    # (A learned belief-conditioned value leaf was implemented and tried -- see
    # collect_value/train_value -- but the value was too coarse and hurt, so the
    # reported configuration uses the distance leaf.)
    value_fn = None
    caps = []
    for i in PE.SEEDS:
        prey_be = load_params(PE.apath("mappo_intent_belief", "prey", i))
        pred_be = load_params(PE.apath("mappo_intent_belief", "pred", i))
        c = run_episode(env, prey_be, pred_be, predict, value_fn, ks, enc_proba,
                        jax.random.PRNGKey(300 + i))
        caps.append(c.mean()); print(f"  seed {i}: learned-model planner = {c.mean():.2f}", flush=True)
    lm_mean, lm_std = float(np.mean(caps)), float(np.std(caps))

    prev = np.load(os.path.join(LOGDIR, "part2_planner.npz"))
    truesim = (float(prev["planner"].mean()), float(prev["planner"].std()))
    react = (float(prev["ref_belief"].mean()), float(prev["ref_belief"].std()))
    oracle = (float(prev["ref_oracle"].mean()), float(prev["ref_oracle"].std()))
    print("\ncaptures/episode:")
    print(f"  reactive belief        : {react[0]:.2f}")
    print(f"  planner (true sim)     : {truesim[0]:.2f}")
    print(f"  planner (LEARNED model): {lm_mean:.2f} +/- {lm_std:.2f}")
    print(f"  oracle (reactive)      : {oracle[0]:.2f}")
    keep = 100 * lm_mean / truesim[0]
    print(f"learned-model planner keeps {keep:.0f}% of the true-sim planner; "
          f"+{100*(lm_mean-react[0])/react[0]:.0f}% vs reactive")

    np.savez(os.path.join(LOGDIR, "part3_learned_planner.npz"),
             learned=np.array(caps), dyn_rmse1=err1, dyn_rmseH=errH,
             truesim=prev["planner"], react=prev["ref_belief"], oracle=prev["ref_oracle"])

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    labels = ["reactive\nbelief", "planner\n(true sim)", "planner\n(learned model)",
              "oracle\n(reactive)"]
    means = [react[0], truesim[0], lm_mean, oracle[0]]
    stds = [react[1], truesim[1], lm_std, oracle[1]]
    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                  color=["#1f7a4e", "#0d6e7a", "#b85c10", "#073b42"])
    ax.set_ylabel("captures / episode")
    ax.set_title(f"Part 3 — planning with a LEARNED dynamics model\n"
                 f"learned-model planner {lm_mean:.2f} keeps {keep:.0f}% of the "
                 f"true-sim ({truesim[0]:.2f}); pos RMSE {err1:.3f}/1-step, {errH:.3f}/{H}-step")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.05, f"{m:.2f}",
                ha="center", fontweight="bold", fontsize=9)
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    out = os.path.join(PLOTDIR, "part3_learned_planner.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
