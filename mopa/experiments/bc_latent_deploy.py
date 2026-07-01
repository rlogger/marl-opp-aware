"""The right way to ask "does the latent help BC": deployed captures, not
one-step action accuracy.

One-step action-matching has a tiny ceiling -- the predator already sees all
positions, so the strategy adds almost nothing to predicting the very next
action (oracle only buys +2.6 pts). But a predator that knows the prey's
strategy can pre-position and intercept, and that compounds over a whole
episode. So we measure CAPTURES.

Headroom by design: vanilla BC here is PLACEMENT-BLIND -- one clone trained on
both placements pooled, so it must hedge. We then give the same clone the
strategy as an extra input and redeploy:

  random  ->  vanilla pooled BC  ->  latent-conditioned BC  ->  oracle (true
  placement) BC  ->  per-placement MAPPO (expert ceiling).

If telling the clone the strategy recovers the captures a pooled clone loses,
the strategy latent has real downstream value -- the meeting's claim, on the
metric that matters.

Outputs: plots/mopa_bc_latent_deploy.png
         logs/MPE_simple_tag_v3/mopa_bc_latent_deploy.npz
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from mopa import legacy
import generate_trajectory_dataset_resources as G
from generate_trajectory_dataset_resources import (
    _make_env, split_teams, _params_path, ActorLogits, HIDDEN)
from jaxmarl.wrappers.baselines import CTRolloutManager, MPELogWrapper, load_params
from mopa.data import standardize, occupancy
from mopa.encoders import train_vae, probe_acc
from mopa.experiments.bc_vs_mappo import set_wp, bc_features, train_bc
from mopa.experiments.bc_latent_sweep import build_range
from mopa.paths import log_path, plot_path

PLACEMENTS = ("circle", "corners")
EP_LEN = 25
OBS = 100        # steps used to build the prey-strategy latent / demos
LAT = 4


def rollout(placement, pred_mode, prey_p, n_eps, rng,
            bc=None, cond=None, mappo_pred=None):
    """First-episode captures with predators by `pred_mode`
    {'mappo','random','bc'}. For 'bc', `bc`=(net,params,mu,sd) and optional
    `cond` (d,) is appended to every predator feature."""
    base = _make_env(placement); teams = split_teams(base)
    env = MPELogWrapper(base); wrapped = CTRolloutManager(env, batch_size=n_eps)
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
        a_prey = net.apply(prey_p, obs[prey_name][:, :max_obs]).argmax(-1)
        acts = {prey_name: a_prey.astype(jnp.int32)}
        cur = np.asarray(st.env_state.p_pos)
        if pred_mode == "bc":
            net_bc, p_bc, mu, sd = bc
            feats = bc_features(cur[:, pred_idxs], cur[:, prey_idx],
                                prev_pr, prev_py)              # (N,3,19)
            if cond is not None:
                c = np.broadcast_to(np.asarray(cond, np.float32),
                                    (n_eps, 3, len(cond)))
                feats = np.concatenate([feats, c], -1)
            fl = ((feats.reshape(-1, feats.shape[-1]) - mu) / sd).astype(np.float32)
            ap = np.asarray(net_bc.apply(p_bc, jnp.asarray(fl))
                            ).argmax(-1).reshape(n_eps, 3)
            for j, a in enumerate(pred_names):
                acts[a] = jnp.asarray(ap[:, j], jnp.int32)
        elif pred_mode == "mappo":
            for a in pred_names:
                acts[a] = net.apply(mappo_pred, obs[a][:, :max_obs]
                                    ).argmax(-1).astype(jnp.int32)
        else:
            for a in pred_names:
                rng, kk = jax.random.split(rng)
                q = jax.random.uniform(kk, (n_eps, action_dim)) - (1 - valid[a]) * 1e10
                acts[a] = q.argmax(-1).astype(jnp.int32)
        prev_pr, prev_py = cur[:, pred_idxs], cur[:, prey_idx]
        rng, ks = jax.random.split(rng)
        obs, st, rew, _, _ = wrapped.batch_step(ks, st, acts)
        caps += np.asarray(rew[pred_names[0]]) / 10.0
    return caps


def main():
    set_wp(); G.NUM_STEPS = OBS
    seeds = (0, 1, 2)

    # ---- pooled demos + per-placement prey-strategy latent ----
    print(f"collecting demos ({OBS}-step rollouts)...")
    pos_list, S_list, A_list, lab_list = [], [], [], []
    prey_ps, mappo_preds = {}, {}
    for li, placement in enumerate(PLACEMENTS):
        prey_ps[placement] = load_params(_params_path("mappo", placement + "_wp", "prey", 0))
        mappo_preds[placement] = [load_params(_params_path("mappo", placement + "_wp", "pred", s))
                                  for s in seeds]
        for s in seeds:
            d = G.rollout_one_checkpoint("mappo", placement + "_wp", s, 150,
                                         placement, jax.random.PRNGKey(s))
            ds = {"prey_pos": d["positions"], "pred_pos": d["pred_positions"],
                  "pred_act": d["pred_actions"]}
            S, A, _ = build_range(ds, 1, OBS)
            n = len(S)
            S_list.append(S); A_list.append(A)
            lab_list.append(np.full(n, li, np.int32))
            pos_list.append((li, d["positions"]))
    S0 = np.concatenate(S_list); A = np.concatenate(A_list)
    lab = np.concatenate(lab_list)

    # unsupervised latent: VAE on occupancy of the pooled prey trajectories
    occ, labocc = [], []
    for li, P in pos_list:
        occ.append(P); labocc.append(np.full(len(P), li, np.int32))
    P = np.concatenate(occ); labocc = np.concatenate(labocc)
    Xc, _, _ = standardize(occupancy(P, 0, OBS))
    z = train_vae(Xc, jax.random.PRNGKey(0), lat=LAT)
    zs = (z - z.mean(0)) / (z.std(0) + 1e-6)
    zplace = np.stack([zs[labocc == li].mean(0) for li in (0, 1)])   # (2, LAT)
    print(f"  latent probe (placement | z): "
          f"{probe_acc(zs, labocc):.2f}   |zplace gap| "
          f"{np.linalg.norm(zplace[0]-zplace[1]):.2f}")
    onehot = np.eye(2, dtype=np.float32)

    # ---- train three clones on the SAME pooled demos ----
    print("training clones (vanilla / latent / oracle)...")
    bc_van = train_bc(S0, A)
    bc_lat = train_bc(np.concatenate([S0, zplace[lab]], -1).astype(np.float32), A)
    bc_orc = train_bc(np.concatenate([S0, onehot[lab]], -1).astype(np.float32), A)

    # ---- deploy, measure captures per placement ----
    res = {k: [] for k in ("random", "vanilla", "latent", "oracle", "mappo")}
    for placement in PLACEMENTS:
        li = PLACEMENTS.index(placement)
        for k, s in enumerate(seeds):
            r = jax.random.PRNGKey(900 + s)
            res["random"].append(rollout(placement, "random", prey_ps[placement], 128, r).mean())
            res["vanilla"].append(rollout(placement, "bc", prey_ps[placement], 128, r, bc=bc_van).mean())
            res["latent"].append(rollout(placement, "bc", prey_ps[placement], 128, r, bc=bc_lat, cond=zplace[li]).mean())
            res["oracle"].append(rollout(placement, "bc", prey_ps[placement], 128, r, bc=bc_orc, cond=onehot[li]).mean())
            res["mappo"].append(rollout(placement, "mappo", prey_ps[placement], 128, r, mappo_pred=mappo_preds[placement][k]).mean())
        print(f"  [{placement}] " + "  ".join(
            f"{k} {np.mean(res[k][-len(seeds):]):.2f}" for k in res))

    M = {k: (float(np.mean(v)), float(np.std(v))) for k, v in res.items()}
    span = M["mappo"][0] - M["vanilla"][0]
    rec = (lambda x: 100 * (x - M["vanilla"][0]) / (span + 1e-9))
    print(f"\noverall captures/ep: " + "  ".join(f"{k} {M[k][0]:.2f}" for k in res))
    print(f"latent recovers {rec(M['latent'][0]):.0f}% and oracle "
          f"{rec(M['oracle'][0]):.0f}% of the pooled-BC -> MAPPO gap")

    np.savez(log_path("mopa_bc_latent_deploy.npz"), **{f"{k}": np.array(res[k]) for k in res})

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    order = ["random", "vanilla", "latent", "oracle", "mappo"]
    labels = ["random", "vanilla BC\n(placement-blind)", "+ latent\nπ(a|s,z)",
              "+ oracle\nπ(a|s,placement)", "MAPPO\n(per-placement)"]
    means = [M[k][0] for k in order]; errs = [M[k][1] for k in order]
    bars = ax.bar(labels, means, yerr=errs, capsize=5,
                  color=["#bbb", "#7f9c9f", "#0d6e7a", "#0a4f58", "#073b42"])
    ax.set_ylabel("captures / episode")
    ax.set_title("Conditioning a placement-blind BC clone on the prey-strategy "
                 "latent recovers deployed captures\n"
                 f"latent recovers {rec(M['latent'][0]):.0f}% of the pooled-BC→MAPPO gap "
                 f"(oracle {rec(M['oracle'][0]):.0f}%)", fontsize=10.5)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.03, f"{m:.2f}",
                ha="center", fontweight="bold", fontsize=9)
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    out = plot_path("mopa_bc_latent_deploy.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
