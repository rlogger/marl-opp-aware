"""Part 4: a JEPA latent world model for the planner (fixing Part 3).

Part 3 learned the dynamics in the raw STATE space (predict pos/vel) and the
planner exploited the model's error on the off-distribution joint actions it
queries -- it recovered only 59% of the true-simulator planner. Here we apply the
same predictive principle that fixed the encoder: predict the next LATENT, not the
next state.

  encoder   h : agent state (16) -> latent s
  dynamics  g : (s, joint action) -> next latent s'   (JEPA: g(h(x),a) ~ h_ema(x'))
  decoder   d : s -> agent state                       (readout for the planner)

The composed  predict = d . g . h  has the SAME signature as Part 3's state-space
model, so it drops straight into Part 3's planner. The question: does predicting
in a smooth, regularised latent generalise better to the planner's OOD actions
than predicting raw pos/vel?

Output: plots/part4_jepa_world_planner.png, logs/.../part4_jepa_world_planner.npz
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal
from flax.training.train_state import TrainState
import optax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import part2_intent_eval as PE
import part3_learned_planner as P3
from jaxmarl.wrappers.baselines import load_params

LOGDIR = "logs/MPE_simple_tag_v3"; PLOTDIR = "plots"
LATD, HID = 48, 128


class Henc(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        return nn.Dense(LATD, kernel_init=orthogonal(1.0))(x)


class Gdyn(nn.Module):
    @nn.compact
    def __call__(self, s, a):
        x = jnp.concatenate([s, a], -1)
        x = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        x = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        return nn.Dense(LATD, kernel_init=orthogonal(1.0))(x)


class Ddec(nn.Module):
    @nn.compact
    def __call__(self, s):
        s = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(s))
        return nn.Dense(16, kernel_init=orthogonal(1.0))(s)


def train_jepa_world(pos, vel, acts, L=4, steps=12000, seed=0):
    """Rollout training in latent space: roll g, decode each step to match the true
    future state, and keep g predicting the EMA-encoded next latent (JEPA)."""
    X = np.concatenate([pos.reshape(*pos.shape[:2], 8), vel.reshape(*vel.shape[:2], 8)], -1)
    mu, sd = X.mean((0, 1)), X.std((0, 1)) + 1e-6
    Xn = (X - mu) / sd                                    # (n, T+1, 16)
    n_envs, T1 = Xn.shape[0], Xn.shape[1]; T = T1 - 1
    h, g, d = Henc(), Gdyn(), Ddec()
    rng = jax.random.PRNGKey(seed); rh, rg, rd = jax.random.split(rng, 3)
    p = {"h": h.init(rh, jnp.zeros((1, 16))),
         "g": g.init(rg, jnp.zeros((1, LATD)), jnp.zeros((1, 20))),
         "d": d.init(rd, jnp.zeros((1, LATD)))}
    tgt = p["h"]; tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-4))
    opt = tx.init(p); Xj, Aj = jnp.asarray(Xn), jnp.asarray(acts); MOM = 0.997

    def loss(p, tgt, ei, t0):
        x0 = Xj[ei, t0]
        aw = jax.nn.one_hot(Aj[ei[:, None], t0[:, None] + jnp.arange(L)], 5)   # (b,L,4,5)
        aw = aw.reshape(aw.shape[0], L, 20)

        def step(carry, l):
            s = carry
            sn = g.apply(p["g"], s, aw[:, l])
            xn_true = Xj[ei, t0 + l + 1]
            tn = jax.lax.stop_gradient(h.apply(tgt, xn_true))
            dec = d.apply(p["d"], sn)
            return sn, (dec, sn, tn, xn_true)
        s0 = h.apply(p["h"], x0)
        _, (decs, sns, tns, xs) = jax.lax.scan(step, s0, jnp.arange(L))
        rec = jnp.mean((decs - xs) ** 2)                  # decode rollout -> true future
        jep = jnp.mean(jnp.abs(sns - tns))                # latent predicts EMA next latent
        var = jnp.mean(jax.nn.relu(1 - jnp.sqrt(sns.var(1) + 1e-4)))
        return rec + 0.1 * jep + 0.02 * var

    @jax.jit
    def upd(p, tgt, opt, ei, t0):
        gr = jax.grad(loss)(p, tgt, ei, t0)
        u, opt = tx.update(gr, opt); p = optax.apply_updates(p, u)
        tgt = jax.tree_util.tree_map(lambda a, b: MOM * a + (1 - MOM) * b, tgt, p["h"])
        return p, tgt, opt
    for s in range(steps):
        rng, k1, k2 = jax.random.split(rng, 3)
        ei = jax.random.randint(k1, (256,), 0, n_envs); t0 = jax.random.randint(k2, (256,), 0, T - L)
        p, tgt, opt = upd(p, tgt, opt, ei, t0)
        if s % 3000 == 0 or s == steps - 1:
            print(f"    jepa-world step {s:5d}  loss {float(loss(p, tgt, ei, t0)):.4f}", flush=True)

    muj, sdj = jnp.asarray(mu), jnp.asarray(sd)

    def predict(apos, avel, acts_oh):
        x = (jnp.concatenate([apos.reshape(*apos.shape[:-2], 8),
                              avel.reshape(*avel.shape[:-2], 8)], -1) - muj) / sdj
        s = g.apply(p["g"], h.apply(p["h"], x), acts_oh.reshape(*acts_oh.shape[:-2], 20))
        xn = d.apply(p["d"], s) * sdj + muj
        return xn[..., :8].reshape(*apos.shape), xn[..., 8:].reshape(*avel.shape)
    return predict


def main():
    os.makedirs(PLOTDIR, exist_ok=True)
    env = PE.make_env(reveal=True)
    print("collecting transitions...")
    pos, vel, acts = P3.collect_seq(env, load_params(PE.apath("mappo_intent_belief", "prey", 0)),
                                    load_params(PE.apath("mappo_intent_belief", "pred", 0)),
                                    n_envs=512, steps=200, rng=jax.random.PRNGKey(0))
    predict = train_jepa_world(pos, vel, acts)

    # closed-loop H-step decode error (held out)
    tp, tv, ta = P3.collect_seq(env, load_params(PE.apath("mappo_intent_belief", "prey", 1)),
                                load_params(PE.apath("mappo_intent_belief", "pred", 1)),
                                n_envs=128, steps=26, rng=jax.random.PRNGKey(99), eps=0.0)
    ap, av = jnp.asarray(tp[:, 0]), jnp.asarray(tv[:, 0]); errs = []
    for hh in range(min(P3.H, ta.shape[1])):
        ap, av = predict(ap, av, jax.nn.one_hot(jnp.asarray(ta[:, hh]), 5))
        errs.append(float(np.sqrt(((np.asarray(ap) - tp[:, hh + 1]) ** 2).mean())))
    print(f"  JEPA-world pos RMSE: 1-step {errs[0]:.4f}, {len(errs)}-step {errs[-1]:.4f}")

    # belief encoder (supervised, same as Part 3 used)
    posb, intb = [], []
    for i in PE.SEEDS:
        Pp, gg = PE.gather_prey_traj(env, load_params(PE.apath("mappo_intent_oracle", "prey", i)),
                                     load_params(PE.apath("mappo_intent_oracle", "pred", i)),
                                     120, jax.random.PRNGKey(i))
        posb.append(Pp); intb.append(gg)
    posb = np.concatenate(posb); intb = np.concatenate(intb)
    ks = [3, 5, 8, 12, 18, 25]; enc_proba = {}
    for k in ks:
        _, _, _, soft = PE.train_encoder_at_k(posb, intb, k); enc_proba[k] = soft

    caps = []
    for i in PE.SEEDS:
        prey_be = load_params(PE.apath("mappo_intent_belief", "prey", i))
        pred_be = load_params(PE.apath("mappo_intent_belief", "pred", i))
        c = P3.run_episode(env, prey_be, pred_be, predict, None, ks, enc_proba,
                           jax.random.PRNGKey(500 + i))
        caps.append(c.mean()); print(f"  seed {i}: JEPA-world planner {c.mean():.2f}", flush=True)
    jm, js = float(np.mean(caps)), float(np.std(caps))

    p3 = np.load(os.path.join(LOGDIR, "part3_learned_planner.npz"))
    p2 = np.load(os.path.join(LOGDIR, "part2_planner.npz"))
    state_space = float(p3["learned"].mean()); truesim = float(p2["planner"].mean())
    react = float(p2["ref_belief"].mean())
    print(f"\n  reactive belief            : {react:.2f}")
    print(f"  Part 3 state-space planner : {state_space:.2f}")
    print(f"  Part 4 JEPA-world planner  : {jm:.2f} +/- {js:.2f}")
    print(f"  planner (true simulator)   : {truesim:.2f}")

    np.savez(os.path.join(LOGDIR, "part4_jepa_world_planner.npz"),
             jepa=np.array(caps), rmse=np.array(errs), state_space=state_space,
             truesim=truesim, react=react)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    labels = ["reactive\nbelief", "Part 3:\nstate-space model", "Part 4:\nJEPA latent model",
              "planner\n(true simulator)"]
    means = [react, state_space, jm, truesim]
    bars = ax.bar(labels, means, yerr=[0, float(p3["learned"].std()), js, float(p2["planner"].std())],
                  capsize=5, color=["#1f7a4e", "#b85c10", "#0d6e7a", "#073b42"])
    ax.set_ylabel("captures / episode")
    keep = 100 * jm / truesim
    ax.set_title(f"Planning through a LEARNED world model: state-space vs JEPA latent\n"
                 f"JEPA-world {jm:.2f} vs state-space {state_space:.2f}; "
                 f"keeps {keep:.0f}% of the true simulator")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.05, f"{m:.2f}", ha="center", fontweight="bold")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(os.path.join(PLOTDIR, "part4_jepa_world_planner.png"), dpi=140); plt.close(fig)
    print("saved plots/part4_jepa_world_planner.png")


if __name__ == "__main__":
    main()
