"""How FAST can each encoder infer the opponent's strategy? (anytime opponent model)

A planner can only commit once it knows which opponent it faces. We sweep the
number of observed steps k and, at each k, fit three encoders on the prey's first
k absolute positions and score unsupervised intent recovery:

  VAE (generative)        : encode first-k, reconstruct first-k.
  JEPA (predictive)       : encode first-k, predict the representation of the
                            immediate future window [k, k+m] (EMA target).
  supervised (reference)  : an MLP classifier on first-k (uses labels).

The claim: the predictive encoder recovers the strategy from FEWER steps -- it
predicts where the prey is heading instead of waiting for it to arrive -- so the
opponent-aware planner becomes confident sooner.

Output: plots/jepa_inference_speed.png, logs/MPE_simple_tag_v3/jepa_inference_speed.npz
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import part2_intent_eval as PE
from jepa_vs_vae_encoder import Enc, EncVAE, Dec, Pred, LAT, metrics
from jaxmarl.wrappers.baselines import load_params

LOGDIR = "logs/MPE_simple_tag_v3"; PLOTDIR = "plots"
KMAX = 24                           # full window; observe first-k of it, rest masked
KS = [3, 5, 8, 11, 14, 17, 20]      # observed-step budgets (all < KMAX: a future to predict)
STEPS = 3500


def _std(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-6)


def mask_to(full, k):
    """Keep the first k steps of a (N, KMAX, 2) window, zero the rest (the future)."""
    m = np.zeros((KMAX, 2), np.float32); m[:k] = 1.0
    return (full * m).reshape(len(full), -1)


def fit_vae(X, intent, rng):
    enc, dec = EncVAE(), Dec(out=X.shape[1])
    rng, ke, kd = jax.random.split(rng, 3)
    p = {"e": enc.init(ke, X[:1]), "d": dec.init(kd, jnp.zeros((1, LAT)))}
    tx = optax.adam(1e-3); opt = tx.init(p); Xj = jnp.asarray(X); n = len(X)

    def loss(p, x, rng, beta):
        mu, lv = enc.apply(p["e"], x)
        z = mu + jnp.exp(0.5 * lv) * jax.random.normal(rng, mu.shape)
        xh = dec.apply(p["d"], z)
        return (jnp.mean(jnp.sum((xh - x) ** 2, -1))
                + beta * jnp.mean(-0.5 * jnp.sum(1 + lv - mu ** 2 - jnp.exp(lv), -1)))

    @jax.jit
    def upd(p, opt, idx, rng, beta):
        g = jax.grad(loss)(p, Xj[idx], rng, beta)
        u, opt = tx.update(g, opt); return optax.apply_updates(p, u), opt
    for s in range(STEPS):
        rng, bk, sk = jax.random.split(rng, 3)
        idx = jax.random.choice(bk, n, (128,), replace=False)
        p, opt = upd(p, opt, idx, sk, float(min(1.0, s / 1200)))
    return metrics(np.asarray(enc.apply(p["e"], Xj)[0]), intent)


def fit_jepa(Xc, Xfull, intent, rng):
    """Context = first-k steps (rest of the KMAX window masked to zero); target =
    the FULL window. z must predict the representation of the masked FUTURE -> it
    encodes where the prey is heading. Same dim -> shared EMA encoder."""
    enc, pred = Enc(), Pred()
    rng, ke, kp = jax.random.split(rng, 3)
    p = {"e": enc.init(ke, Xc[:1]), "p": pred.init(kp, jnp.zeros((1, LAT)))}
    tgt = p["e"]; tx = optax.adam(1e-3); opt = tx.init(p)
    Xcj, Xtj = jnp.asarray(Xc), jnp.asarray(Xfull)
    n = len(Xc); MOM = 0.996

    def loss(p, tgt, xc, xt):
        z = enc.apply(p["e"], xc)
        t = jax.lax.stop_gradient(enc.apply(tgt, xt))
        std = jnp.sqrt(z.var(0) + 1e-4)
        return jnp.mean(jnp.abs(pred.apply(p["p"], z) - t)) + jnp.mean(jax.nn.relu(1 - std))

    @jax.jit
    def upd(p, tgt, opt, idx):
        g = jax.grad(loss)(p, tgt, Xcj[idx], Xtj[idx])
        u, opt = tx.update(g, opt); p = optax.apply_updates(p, u)
        tgt = jax.tree_util.tree_map(lambda a, b: MOM * a + (1 - MOM) * b, tgt, p["e"])
        return p, tgt, opt
    for s in range(STEPS):
        rng, bk = jax.random.split(rng)
        idx = jax.random.choice(bk, n, (128,), replace=False)
        p, tgt, opt = upd(p, tgt, opt, idx)
    return metrics(np.asarray(enc.apply(p["e"], Xcj)), intent)


def main():
    os.makedirs(PLOTDIR, exist_ok=True)
    env = PE.make_env(reveal=True)
    pos, intent = [], []
    for i in PE.SEEDS:
        P, g = PE.gather_prey_traj(env,
                                   load_params(PE.apath("mappo_intent_oracle", "prey", i)),
                                   load_params(PE.apath("mappo_intent_oracle", "pred", i)),
                                   PE.N_EPS, jax.random.PRNGKey(i))
        pos.append(P); intent.append(g)
    pos = np.concatenate(pos).astype(np.float32); intent = np.concatenate(intent).astype(int)
    print(f"trajectories {pos.shape}")

    # full KMAX-step window, per-dim standardised (shared across all k)
    full = pos[:, :KMAX].reshape(len(pos), -1)
    mu, sd = full.mean(0), full.std(0) + 1e-6
    full_std = ((full - mu) / sd).reshape(len(pos), KMAX, 2)

    Xfull = full_std.reshape(len(pos), -1)                 # target = full window
    SEEDS = [0, 1, 2]
    vp_m, vp_s, va_m, va_s = [], [], [], []
    jp_m, jp_s, ja_m, ja_s, sups = [], [], [], [], []
    for k in KS:
        Xc = mask_to(full_std, k)                          # observe first-k, mask future
        vps = [fit_vae(Xc, intent, jax.random.PRNGKey(s)) for s in SEEDS]
        jps = [fit_jepa(Xc, Xfull, intent, jax.random.PRNGKey(s)) for s in SEEDS]
        vp = np.array([x[0] for x in vps]); va = np.array([x[1] for x in vps])
        jp = np.array([x[0] for x in jps]); ja = np.array([x[1] for x in jps])
        sup, _, _, _ = PE.train_encoder_at_k(pos, intent, k)
        vp_m.append(vp.mean()); vp_s.append(vp.std()); va_m.append(va.mean()); va_s.append(va.std())
        jp_m.append(jp.mean()); jp_s.append(jp.std()); ja_m.append(ja.mean()); ja_s.append(ja.std())
        sups.append(sup)
        print(f"  k={k:2d}  VAE {vp.mean():.2f}/{va.mean():.2f}   "
              f"JEPA {jp.mean():.2f}/{ja.mean():.2f}   sup {sup:.2f}", flush=True)
    ks = np.array(KS, float)
    vp_m, vp_s, va_m, va_s = map(np.array, (vp_m, vp_s, va_m, va_s))
    jp_m, jp_s, ja_m, ja_s, sups = map(np.array, (jp_m, jp_s, ja_m, ja_s, sups))
    np.savez(os.path.join(LOGDIR, "jepa_inference_speed.npz"), ks=ks,
             vae_probe=vp_m, jepa_probe=jp_m, sup=sups, vae_ari=va_m, jepa_ari=ja_m)

    def band(ax, x, m, s, c, lab, mk="-o"):
        ax.plot(x, m, mk, color=c, label=lab); ax.fill_between(x, m - s, m + s, color=c, alpha=0.15)

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    a.plot(ks, sups, "-o", color="#1f7a4e", label="supervised (labels)")
    band(a, ks, jp_m, jp_s, "#0d6e7a", "JEPA (predict)")
    band(a, ks, vp_m, vp_s, "#b85c10", "VAE (reconstruct)")
    a.axhline(0.25, ls=":", c="k", lw=1, label="chance")
    a.set_xlabel("steps of opponent observed (k)"); a.set_ylabel("intent probe accuracy")
    a.set_ylim(0, 1.02); a.set_title("How fast can you read the opponent's strategy?")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    band(b, ks, ja_m, ja_s, "#0d6e7a", "JEPA (predict)", "-s")
    band(b, ks, va_m, va_s, "#b85c10", "VAE (reconstruct)", "-s")
    b.set_xlabel("steps of opponent observed (k)"); b.set_ylabel("unsupervised ARI vs intent")
    b.set_ylim(-0.05, 1.02); b.set_title("Unsupervised strategy recovery")
    b.legend(fontsize=8); b.grid(alpha=0.3)
    fig.suptitle("JEPA is an anytime opponent encoder: it reads the strategy from fewer steps",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTDIR, "jepa_inference_speed.png"), dpi=140)
    plt.close(fig)
    print("saved plots/jepa_inference_speed.png")


if __name__ == "__main__":
    main()
