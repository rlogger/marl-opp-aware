"""JEPA vs VAE: which self-supervised encoder recovers a hidden opponent strategy?

Part 1 of the proposal needs an encoder q(z | tau^red) whose latent carries the
opponent's *strategy*. We compare two self-supervised objectives on the SAME
opponent (prey) trajectories from the hidden-intent task, with NO intent labels:

  VAE  (generative)  : encode the first 12 steps, then RECONSTRUCT them (ELBO).
                       Spends capacity on every detail, including the random start
                       and the evasion noise.
  JEPA (predictive)  : encode the first 12 steps, then PREDICT the *representation*
                       of the future window (steps 12-24, the corner-approach) via
                       an EMA target encoder. No reconstruction -- it keeps only
                       the predictable structure (where the prey is heading = its
                       intent) and discards the noise.

Both use a 2-D latent (so the latent space is the figure, no PCA). We score how
well each latent recovers the 4 ground-truth corner intents, unsupervised:
  - probe : 4-way logistic-regression CV accuracy on z      (chance 0.25)
  - ARI   : GMM(k=4) on z vs the true intent
against a supervised ceiling (an MLP classifier on the raw window).

Outputs:
  plots/jepa_vs_vae_encoder.png    final latents + metric bars
  plots/jepa_vs_vae_training.gif   the two latent spaces forming during training
  logs/MPE_simple_tag_v3/jepa_vs_vae.npz
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
import optax

from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import part2_intent_eval as PE
from exp4_vae_separation import probe_acc
from jaxmarl.wrappers.baselines import load_params

LOGDIR = "logs/MPE_simple_tag_v3"; PLOTDIR = "plots"
K = 4                       # intents
CTX, TGT = (0, 12), (12, 24)   # context / target windows (steps)
LAT = 2                     # latent dim (== the figure)
HID = 64
STEPS = 5000
SNAP_EVERY = 320
INT_COL = ["#0d6e7a", "#b85c10", "#1f7a4e", "#b44233"]


# --------------------------------------------------------------------------- #
class Enc(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        x = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        return nn.Dense(LAT, kernel_init=orthogonal(1.0))(x)


class EncVAE(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        x = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        return nn.Dense(LAT)(x), nn.Dense(LAT)(x)        # mu, logvar


class Dec(nn.Module):
    out: int
    @nn.compact
    def __call__(self, z):
        z = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(z))
        z = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(z))
        return nn.Dense(self.out)(z)


class Pred(nn.Module):
    @nn.compact
    def __call__(self, z):
        z = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(z))
        return nn.Dense(LAT, kernel_init=orthogonal(1.0))(z)


def metrics(z, intent):
    zc = (z - z.mean(0)) / (z.std(0) + 1e-6)
    probe = probe_acc(zc, intent)
    gm = GaussianMixture(K, n_init=8, random_state=0).fit(zc)
    ari = adjusted_rand_score(intent, gm.predict(zc))
    return probe, ari


# --------------------------------------------------------------------------- #
def train_vae(Xc, intent, rng):
    enc, dec = EncVAE(), Dec(out=Xc.shape[1])
    rng, ke, kd = jax.random.split(rng, 3)
    pe = enc.init(ke, Xc[:1]); pd = dec.init(kd, jnp.zeros((1, LAT)))
    params = {"e": pe, "d": pd}
    tx = optax.adam(1e-3); opt = tx.init(params)
    Xj = jnp.asarray(Xc); n = len(Xc)

    def loss_fn(p, x, rng, beta):
        mu, lv = enc.apply(p["e"], x)
        z = mu + jnp.exp(0.5 * lv) * jax.random.normal(rng, mu.shape)
        xh = dec.apply(p["d"], z)
        rec = jnp.mean(jnp.sum((xh - x) ** 2, -1))
        kl = jnp.mean(-0.5 * jnp.sum(1 + lv - mu ** 2 - jnp.exp(lv), -1))
        return rec + beta * kl

    @jax.jit
    def upd(params, opt, idx, rng, beta):
        g = jax.grad(loss_fn)(params, Xj[idx], rng, beta)
        u, opt = tx.update(g, opt); return optax.apply_updates(params, u), opt

    snaps = []
    for s in range(STEPS):
        rng, bk, sk = jax.random.split(rng, 3)
        idx = jax.random.choice(bk, n, (128,), replace=False)
        beta = float(min(1.0, s / 1500))
        params, opt = upd(params, opt, idx, sk, beta)
        if s % SNAP_EVERY == 0 or s == STEPS - 1:
            z = np.asarray(enc.apply(params["e"], Xj)[0])
            snaps.append((s, z, *metrics(z, intent)))
    return snaps


def train_jepa(Xc, Xt, intent, rng):
    enc, pred = Enc(), Pred()
    rng, ke, kp = jax.random.split(rng, 3)
    pe = enc.init(ke, Xc[:1]); pp = pred.init(kp, jnp.zeros((1, LAT)))
    params = {"e": pe, "p": pp}
    target = pe                                  # EMA target encoder (anti-collapse)
    tx = optax.adam(1e-3); opt = tx.init(params)
    Xcj, Xtj = jnp.asarray(Xc), jnp.asarray(Xt); n = len(Xc)
    M = 0.996

    def loss_fn(p, tgt_params, xc, xt):
        z = enc.apply(p["e"], xc)
        pz = pred.apply(p["p"], z)
        t = jax.lax.stop_gradient(enc.apply(tgt_params, xt))     # EMA target, no grad
        predict = jnp.mean(jnp.abs(pz - t))
        # VICReg variance term: keep per-dim std near 1 so z can't collapse
        std = jnp.sqrt(z.var(0) + 1e-4)
        var = jnp.mean(jax.nn.relu(1.0 - std))
        return predict + 1.0 * var

    @jax.jit
    def upd(params, target, opt, idx):
        g = jax.grad(loss_fn)(params, target, Xcj[idx], Xtj[idx])
        u, opt = tx.update(g, opt); params = optax.apply_updates(params, u)
        target = jax.tree_util.tree_map(lambda a, b: M * a + (1 - M) * b,
                                        target, params["e"])
        return params, target, opt

    snaps = []
    for s in range(STEPS):
        rng, bk = jax.random.split(rng)
        idx = jax.random.choice(bk, n, (128,), replace=False)
        params, target, opt = upd(params, target, opt, idx)
        if s % SNAP_EVERY == 0 or s == STEPS - 1:
            z = np.asarray(enc.apply(params["e"], Xcj))
            snaps.append((s, z, *metrics(z, intent)))
    return snaps


# --------------------------------------------------------------------------- #
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
    print(f"trajectories: {pos.shape}, intents {np.bincount(intent)}")

    Xc = pos[:, CTX[0]:CTX[1]].reshape(len(pos), -1)
    Xt = pos[:, TGT[0]:TGT[1]].reshape(len(pos), -1)
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-6)
    Xt = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-6)

    # supervised ceiling: MLP classifier on the raw context window
    sup, _, _, _ = PE.train_encoder_at_k(pos, intent, CTX[1])

    print("training VAE (generative)...");  vae = train_vae(Xc, intent, jax.random.PRNGKey(0))
    print("training JEPA (predictive)..."); jepa = train_jepa(Xc, Xt, intent, jax.random.PRNGKey(0))
    vp, va = vae[-1][2], vae[-1][3]
    jp, ja = jepa[-1][2], jepa[-1][3]
    print(f"\n  supervised ceiling (probe): {sup:.2f}   chance 0.25")
    print(f"  VAE  (generative) : probe {vp:.2f}  ARI {va:.2f}")
    print(f"  JEPA (predictive) : probe {jp:.2f}  ARI {ja:.2f}")

    np.savez(os.path.join(LOGDIR, "jepa_vs_vae.npz"),
             sup=sup, vae_probe=vp, vae_ari=va, jepa_probe=jp, jepa_ari=ja,
             vae_z=vae[-1][1], jepa_z=jepa[-1][1], intent=intent)

    # ---- static comparison figure ---- #
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a, (name, snap, sub) in zip(ax[:2],
            [("VAE  (generative — reconstructs the trajectory)", vae, "reconstruction"),
             ("JEPA  (predictive — predicts the future representation)", jepa, "prediction")]):
        z = snap[-1][1]
        for k in range(K):
            m = intent == k
            a.scatter(z[m, 0], z[m, 1], s=10, alpha=0.6, c=INT_COL[k], label=f"intent {k}")
        a.set_title(f"{name}\nprobe {snap[-1][2]:.2f}   ARI {snap[-1][3]:.2f}", fontsize=10.5)
        a.set_xlabel("latent dim 1"); a.set_ylabel("latent dim 2"); a.legend(fontsize=7)
    b = ax[2]
    labels = ["VAE\nprobe", "JEPA\nprobe", "VAE\nARI", "JEPA\nARI"]
    vals = [vp, jp, va, ja]; cols = ["#999", "#0d6e7a", "#999", "#0d6e7a"]
    bars = b.bar(labels, vals, color=cols)
    b.axhline(sup, ls="--", c="#1f7a4e", lw=1.4, label=f"supervised ceiling {sup:.2f}")
    b.axhline(0.25, ls=":", c="k", lw=1, label="chance (probe) 0.25")
    b.set_ylim(0, 1.02); b.set_ylabel("intent recovery (unsupervised)")
    b.set_title("Which latent carries the strategy?"); b.legend(fontsize=8)
    for bar, v in zip(bars, vals):
        b.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)
    fig.suptitle("Self-supervised opponent encoders, head to head: predict (JEPA) vs reconstruct (VAE)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTDIR, "jepa_vs_vae_encoder.png"), dpi=140); plt.close(fig)
    print("saved plots/jepa_vs_vae_encoder.png")

    # ---- training-progress GIF: the two latent spaces forming ---- #
    nfr = min(len(vae), len(jepa))
    figg, axg = plt.subplots(1, 2, figsize=(10.2, 5.2))

    def lims(snaps):
        allz = np.concatenate([s[1] for s in snaps]);
        return allz.min(0) - 0.3, allz.max(0) + 0.3
    vlo, vhi = lims(vae); jlo, jhi = lims(jepa)

    def draw(fr):
        for a in axg: a.clear()
        for a, snaps, name, lo, hi in [(axg[0], vae, "VAE  (reconstruct)", vlo, vhi),
                                       (axg[1], jepa, "JEPA  (predict)", jlo, jhi)]:
            s, z, p, ar = snaps[fr]
            for k in range(K):
                m = intent == k
                a.scatter(z[m, 0], z[m, 1], s=9, alpha=0.6, c=INT_COL[k])
            a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1])
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{name}\nstep {s}   probe {p:.2f}   ARI {ar:.2f}", fontsize=11)
        figg.suptitle("Watch the opponent's strategy separate: JEPA (predict) vs VAE (reconstruct)",
                      fontweight="bold")
    ani = animation.FuncAnimation(figg, draw, frames=nfr, interval=450)
    ani.save(os.path.join(PLOTDIR, "jepa_vs_vae_training.gif"), writer="pillow", fps=2, dpi=110)
    plt.close(figg)
    print("saved plots/jepa_vs_vae_training.gif")


if __name__ == "__main__":
    main()
