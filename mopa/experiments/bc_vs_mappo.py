"""6/12 next step: compare BC performance with MAPPO performance.

The MAPPO predator is the expert. Vanilla behaviour cloning imitates it from a
hand-built state feature (all agent positions + a one-step velocity proxy +
predator id) -- strictly less information than the policy network sees. We ask
two questions:

  1. action match  -- how often does the cloned policy pick the action the MAPPO
     predator actually took (held-out)?
  2. deployed task performance -- if we DEPLOY the cloned predators in the env
     against the same prey, how many captures/episode do they get, vs the MAPPO
     predators they cloned (and a random-action floor)?

This establishes the vanilla-BC baseline the next step builds on: the gap from
BC to MAPPO is the performance a strategy latent stands to recover.

Outputs: plots/mopa_bc_vs_mappo.png
         logs/MPE_simple_tag_v3/mopa_bc_vs_mappo.npz
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

from mopa import legacy
import generate_trajectory_dataset_resources as G
from generate_trajectory_dataset_resources import (
    _make_env, split_teams, _params_path, ActorLogits, HIDDEN)
from jaxmarl.wrappers.baselines import CTRolloutManager, MPELogWrapper, load_params
from mopa.bc import BCNet, build_samples, BC_BATCH, BC_STEPS
from mopa.paths import log_path, plot_path
from mopa.samples import predator_state_features
from mopa.splits import episode_validation_mask

PLACEMENTS = ("circle", "corners")
EP_LEN = 25
N_PRED = 3


def set_wp():
    G.PRED_MAX_SPEED, G.PRED_ACCEL, G.COLLECT_REWARD = 0.6, 1.5, 10.0


def bc_features(preds, prey, prev_preds, prev_prey):
    """preds (N,3,2), prey (N,2) at step t plus previous -> (N,3,19) matching
    mopa.bc.build_samples: [positions(8), velocity(8), predator-id(3)]."""
    return predator_state_features(preds, prey, prev_preds, prev_prey)


def train_bc(S, A, seed=0, steps=BC_STEPS):
    mu, sd = S.mean(0), S.std(0) + 1e-6
    Sn = (S - mu) / sd
    net = BCNet()
    key = jax.random.PRNGKey(seed)
    key, ki = jax.random.split(key)
    params = net.init(ki, Sn[:1])
    tx = optax.adam(1e-3); opt = tx.init(params)
    Sj, Aj = jnp.asarray(Sn), jnp.asarray(A); n = len(Sn)

    def loss_fn(p, s, a):
        return optax.softmax_cross_entropy_with_integer_labels(
            net.apply(p, s), a).mean()

    @jax.jit
    def upd(params, opt, idx):
        g = jax.grad(loss_fn)(params, Sj[idx], Aj[idx])
        u, opt = tx.update(g, opt)
        return optax.apply_updates(params, u), opt

    for _ in range(steps):
        key, bk = jax.random.split(key)
        idx = jax.random.choice(bk, n, (min(BC_BATCH, n),), replace=False)
        params, opt = upd(params, opt, idx)
    return net, params, mu, sd


def rollout(placement, pred_mode, prey_params, n_eps, rng,
            bc=None, mappo_pred=None):
    """One first-episode rollout; predators act by `pred_mode` in
    {'mappo','bc','random'}. Returns captures/episode (n_eps,)."""
    base = _make_env(placement)
    teams = split_teams(base)
    env = MPELogWrapper(base)
    wrapped = CTRolloutManager(env, batch_size=n_eps)
    prey_name = teams["prey"][0]; pred_names = list(teams["pred"])
    prey_idx = base.agents.index(prey_name)
    pred_idxs = [base.agents.index(n) for n in pred_names]
    action_dim = wrapped.max_action_space
    max_obs = max(base.observation_space(a).shape[0] for a in base.agents)
    net = ActorLogits(action_dim=action_dim, hidden_dim=HIDDEN)

    rng, kr = jax.random.split(rng)
    obs, st = wrapped.batch_reset(kr)
    pp = np.asarray(st.env_state.p_pos)
    prev_pr, prev_py = pp[:, pred_idxs], pp[:, prey_idx]
    caps = np.zeros(n_eps)
    for t in range(EP_LEN):
        valid = wrapped.get_valid_actions(st.env_state)
        a_prey = net.apply(prey_params, obs[prey_name][:, :max_obs]).argmax(-1)
        acts = {prey_name: a_prey.astype(jnp.int32)}

        cur = np.asarray(st.env_state.p_pos)
        if pred_mode == "bc":
            net_bc, p_bc, mu, sd = bc
            feats = bc_features(cur[:, pred_idxs], cur[:, prey_idx],
                                prev_pr, prev_py)                    # (N,3,19)
            fl = ((feats.reshape(-1, 19) - mu) / sd).astype(np.float32)
            ap = np.asarray(net_bc.apply(p_bc, jnp.asarray(fl))
                            ).argmax(-1).reshape(n_eps, 3)
            for j, a in enumerate(pred_names):
                acts[a] = jnp.asarray(ap[:, j], jnp.int32)
        elif pred_mode == "mappo":
            for a in pred_names:
                q = net.apply(mappo_pred, obs[a][:, :max_obs])
                acts[a] = q.argmax(-1).astype(jnp.int32)
        else:                                                       # random
            for a in pred_names:
                rng, kk = jax.random.split(rng)
                q = jax.random.uniform(kk, (n_eps, action_dim))
                q = q - (1 - valid[a]) * 1e10
                acts[a] = q.argmax(-1).astype(jnp.int32)

        prev_pr, prev_py = cur[:, pred_idxs], cur[:, prey_idx]
        rng, ks = jax.random.split(rng)
        obs, st, rew, _, _ = wrapped.batch_step(ks, st, acts)
        caps += np.asarray(rew[pred_names[0]]) / 10.0
    return caps


def main():
    set_wp()
    seeds = (0, 1, 2)
    res = {m: {"circle": [], "corners": []}
           for m in ("mappo", "bc", "random")}
    match = {"circle": [], "corners": []}

    for placement in PLACEMENTS:
        prey_p = load_params(_params_path("mappo", placement + "_wp", "prey", 0))
        pred_ps = [load_params(_params_path("mappo", placement + "_wp", "pred", s))
                   for s in seeds]

        # demos for BC: roll out the MAPPO predators, collect (state, action)
        print(f"[{placement}] collecting MAPPO demos...")
        S_all, A_all, ep_all = [], [], []
        for s in seeds:
            d = G.rollout_one_checkpoint("mappo", placement + "_wp", s, 200,
                                         placement, jax.random.PRNGKey(s),
                                         log_obs=False)
            ds = {"prey_pos": d["positions"], "pred_pos": d["pred_positions"],
                  "pred_act": d["pred_actions"]}
            S, A, ep = build_samples(ds, ctx=1)
            S_all.append(S); A_all.append(A)
            ep_all.append(ep + s * 100000)              # unique episode ids per seed
        S = np.concatenate(S_all); A = np.concatenate(A_all)
        ep = np.concatenate(ep_all)

        # held-out action match -- EPISODE-LEVEL split (all steps of an episode
        # in one fold, so highly-autocorrelated timesteps don't leak). The same
        # train split trains the clone deployed below.
        vmask = episode_validation_mask(ep, rng_seed=0, val_frac=0.2)
        net_bc, p_bc, mu, sd = train_bc(S[~vmask], A[~vmask])
        pred = np.asarray(net_bc.apply(p_bc, jnp.asarray((S[vmask] - mu) / sd))
                          ).argmax(-1)
        match[placement].append(float((pred == A[vmask]).mean()))

        # deployed captures, per checkpoint seed
        for k, s in enumerate(seeds):
            rng = jax.random.PRNGKey(500 + s)
            res["mappo"][placement].append(
                rollout(placement, "mappo", prey_p, 128, rng,
                        mappo_pred=pred_ps[k]).mean())
            res["bc"][placement].append(
                rollout(placement, "bc", prey_p, 128, rng,
                        bc=(net_bc, p_bc, mu, sd)).mean())
            res["random"][placement].append(
                rollout(placement, "random", prey_p, 128, rng).mean())
        print(f"  [{placement}] match {match[placement][-1]:.3f}  "
              f"mappo {np.mean(res['mappo'][placement]):.2f}  "
              f"bc {np.mean(res['bc'][placement]):.2f}  "
              f"random {np.mean(res['random'][placement]):.2f}")

    def ms(m):
        v = np.array(res[m]["circle"] + res[m]["corners"])
        return float(v.mean()), float(v.std())
    mappo, bc, rand = ms("mappo"), ms("bc"), ms("random")
    keep = 100 * (bc[0] - rand[0]) / (mappo[0] - rand[0] + 1e-9)
    print(f"\noverall  mappo {mappo[0]:.2f}  bc {bc[0]:.2f}  random {rand[0]:.2f}"
          f"   BC recovers {keep:.0f}% of MAPPO-over-random; "
          f"action match {np.mean(match['circle']+match['corners']):.3f}")

    np.savez(log_path("mopa_bc_vs_mappo.npz"),
             **{f"{m}_{p}": np.array(res[m][p])
                for m in res for p in PLACEMENTS},
             match=np.array(match["circle"] + match["corners"]))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    labels = ["random\npredators", "vanilla BC\nπ(a|s)", "MAPPO\n(expert)"]
    means = [rand[0], bc[0], mappo[0]]
    errs = [rand[1], bc[1], mappo[1]]
    bars = ax.bar(labels, means, yerr=errs, capsize=5,
                  color=["#bbb", "#0d6e7a", "#073b42"])
    ax.set_ylabel("captures / episode")
    ax.set_title("Behaviour cloning vs the MAPPO expert it imitates\n"
                 f"vanilla BC recovers {keep:.0f}% of the expert's edge over "
                 f"random (held-out action match {np.mean(match['circle']+match['corners']):.2f})")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.03, f"{m:.2f}",
                ha="center", fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = plot_path("mopa_bc_vs_mappo.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
